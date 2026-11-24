"""Experiment stages. The notebook and scripts/train.py call the same functions, so results match."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate,
                                     learning_curve)

from . import config as cfg
from . import pipeline as pl

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
SCORING = {"AP": "average_precision", "ROC": "roc_auc", "Brier": "neg_brier_score"}

PROTOCOL = """Decision rules fixed BEFORE any model result was seen
1. The test set (20%) is used once, after model, strategy and threshold are fixed.
2. Primary metric: average precision (AP) - it focuses on the minority class. ROC-AUC, Brier, F1 etc. are secondary.
3. Hyperparameters: 5-fold stratified CV on the training set (60%), maximising AP.
4. Imbalance strategy: keep 'None' unless another strategy raises mean CV AP by more than 0.005
   (a smaller gain does not justify distorted probabilities).
5. Model: among models whose CV AP is within one standard error of the best, take the simplest
   (Logistic Regression < Decision Tree < Random Forest < Gradient Boosting).
6. Threshold: the value maximising F1 on the validation set (20%).
7. Gender and race are excluded as inputs on ethical grounds; their effect is measured, not exploited."""

GRIDS = {
    "Logistic Regression": {"clf__C": np.logspace(-3, 1, 9).round(5).tolist(), "clf__l1_ratio": [0.0, 0.5, 1.0]},
    "Decision Tree": {"clf__max_depth": [3, 4, 5, 6, 8, 10, None], "clf__min_samples_leaf": [1, 10, 25, 50, 100, 200],
                      "clf__criterion": ["gini", "log_loss"], "clf__ccp_alpha": [0.0, 0.0005, 0.001, 0.002]},
    "Random Forest": {"clf__max_depth": [6, 8, 10, 14, None], "clf__min_samples_leaf": [1, 3, 5, 10, 20],
                      "clf__max_features": ["sqrt", 0.2, 0.35, 0.5]},
    "Gradient Boosting": {"clf__learning_rate": [0.02, 0.05, 0.1], "clf__max_iter": [100, 200, 400],
                          "clf__max_leaf_nodes": [4, 8, 15, 31], "clf__min_samples_leaf": [20, 50, 100],
                          "clf__l2_regularization": [0.0, 1.0, 10.0]},
}
N_ITER = {"Decision Tree": 60, "Random Forest": 25, "Gradient Boosting": 30}


def save(df, name, index=False):
    df.to_csv(cfg.TABLES / f"{name}.csv", index=index, encoding="utf-8")
    return df


def cv_scores(pipe, X, y, train_score=False):
    r = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, return_train_score=train_score, n_jobs=1)
    out = {"CV AP": r["test_AP"].mean(), "CV AP SE": r["test_AP"].std(ddof=1) / np.sqrt(len(r["test_AP"])),
           "CV ROC-AUC": r["test_ROC"].mean(), "CV Brier": -r["test_Brier"].mean()}
    if train_score:
        out["Train AP"] = r["train_AP"].mean()
        out["Gap (train-CV)"] = out["Train AP"] - out["CV AP"]
    return out


# 1 ------------------------------------------------------------------ feature-set ablation
def ablation(train):
    """Which information helps? Logistic Regression with default settings, 5-fold CV on training rows."""
    y = train[cfg.TARGET].to_numpy()
    sets = {
        "Demographics only": (cfg.GROUPS["Demographic"], False),
        "+ Income": (cfg.GROUPS["Demographic"] + cfg.GROUPS["Income"], False),
        "+ Financial well-being": (sum([cfg.GROUPS[g] for g in ["Demographic", "Income", "Financial well-being"]], []), False),
        "+ Resilience": (sum([cfg.GROUPS[g] for g in ["Demographic", "Income", "Financial well-being", "Resilience"]], []), False),
        "+ Credit and banking (all 26)": (cfg.FEATURES, False),
        "All 26 + engineered features": (cfg.FEATURES, True),
        "All + engineered + gender and race (audit only)": (cfg.FEATURES + cfg.PROTECTED, True),
    }
    rows = []
    for label, (feats, derive) in sets.items():
        rows.append({"Feature set": label, "Features": len(feats) + (len(cfg.DERIVED) if derive else 0),
                     **cv_scores(pl.build("Logistic Regression", features=feats, derive=derive), train[feats], y)})
    return save(pd.DataFrame(rows), "ablation")


# 2 ------------------------------------------------------------------ preprocessing choices
def preprocessing_choices(train):
    X, y = pl.xy(train)
    rows = []
    for label, kw in {
        "Credit score: 'Not reported' category, scaled": {},
        "Credit score: most-frequent imputation, scaled": {"credit_missing": "mode"},
        "Credit score: 'Not reported' category, not scaled": {"scale": False},
        "No engineered features": {"derive": False},
    }.items():
        for name in ["Logistic Regression", "Random Forest"]:
            p = {"max_depth": 10, "min_samples_leaf": 5} if name == "Random Forest" else {}
            rows.append({"Choice": label, "Model": name, **cv_scores(pl.build(name, p, **kw), X, y)})
    return save(pd.DataFrame(rows), "preprocessing_choices")


# 3 ------------------------------------------------------------------ tuning
def tune(train):
    X, y = pl.xy(train)
    rows, best, all_cv = [], {}, []
    rows.append({"Model": "Dummy", "Settings": "baseline", **cv_scores(pl.build("Dummy"), X, y, True)})
    for name in pl.MODEL_NAMES:
        rows.append({"Model": name, "Settings": "default", **cv_scores(pl.build(name), X, y, True)})
        kw = dict(scoring=SCORING, refit="AP", cv=CV, return_train_score=True, n_jobs=1 if name == "Random Forest" else -1)
        if name == "Logistic Regression":
            s = GridSearchCV(pl.build(name), GRIDS[name], **kw)
        else:
            s = RandomizedSearchCV(pl.build(name), GRIDS[name], n_iter=N_ITER[name], random_state=cfg.SEED, **kw)
        s.fit(X, y)
        best[name] = {k.replace("clf__", ""): v for k, v in s.best_params_.items()}
        r = pd.DataFrame(s.cv_results_)
        r.insert(0, "Model", name)
        all_cv.append(r[["Model", "params", "mean_test_AP", "std_test_AP", "mean_train_AP", "mean_test_ROC", "mean_test_Brier"]])
        rows.append({"Model": name, "Settings": "tuned", **cv_scores(pl.build(name, best[name]), X, y, True)})
    save(pd.concat(all_cv), "tuning_all_candidates")
    (cfg.MODELS / "best_params.json").write_text(json.dumps(best, indent=2))
    return save(pd.DataFrame(rows), "tuning_summary"), best


# 4 ------------------------------------------------------------------ imbalance strategies
def imbalance(train, valid, best):
    X, y = pl.xy(train)
    Xv, yv = pl.xy(valid)
    rows = []
    for name in pl.MODEL_NAMES:
        for strat in pl.STRATEGIES:
            pipe = pl.build(name, best[name], strat)
            cvs = cv_scores(pipe, X, y)
            p = pipe.fit(X, y).predict_proba(Xv)[:, 1]
            t = best_threshold(yv, p)
            rows.append({"Model": name, "Strategy": strat, **cvs,
                         "Valid AP": average_precision_score(yv, p), "Valid Brier": brier_score_loss(yv, p),
                         "Mean predicted": p.mean(), "Actual rate": yv.mean(),
                         "Recall @0.50": recall_score(yv, p >= .5), "Precision @0.50": precision_score(yv, p >= .5, zero_division=0),
                         "F1 @0.50": f1_score(yv, p >= .5, zero_division=0), "Best-F1 threshold": t,
                         "Recall @best": recall_score(yv, p >= t), "Precision @best": precision_score(yv, p >= t),
                         "F1 @best": f1_score(yv, p >= t)})
    return save(pd.DataFrame(rows), "imbalance_experiments")


def choose(imb):
    """Apply rules 4 and 5 of the protocol."""
    strategies = {}
    for name, g in imb.groupby("Model", sort=False):
        base = g[g.Strategy == "None"].iloc[0]
        top = g.loc[g["CV AP"].idxmax()]
        strategies[name] = top["Strategy"] if top["CV AP"] - base["CV AP"] > 0.005 else "None"
    rows = [imb[(imb.Model == n) & (imb.Strategy == s)].iloc[0] for n, s in strategies.items()]
    t = pd.DataFrame(rows)[["Model", "Strategy", "CV AP", "CV AP SE", "CV ROC-AUC", "CV Brier", "Valid AP", "Valid Brier"]]
    b = t.loc[t["CV AP"].idxmax()]
    t["Within 1 SE of best"] = t["CV AP"] >= b["CV AP"] - b["CV AP SE"]
    order = {n: i for i, n in enumerate(pl.MODEL_NAMES)}
    final = min(t[t["Within 1 SE of best"]]["Model"], key=order.get)
    t["Selected"] = t["Model"].eq(final)
    save(t, "model_selection")
    return t, strategies, final


# 5 ------------------------------------------------------------------ threshold (validation only)
GRID_T = np.round(np.arange(0.02, 0.91, 0.01), 2)


def threshold_table(y, p):
    rows = []
    for t in GRID_T:
        hat = p >= t
        rows.append({"Threshold": t, "Precision": precision_score(y, hat, zero_division=0), "Recall": recall_score(y, hat),
                     "F1": f1_score(y, hat, zero_division=0), "Accuracy": accuracy_score(y, hat),
                     "Balanced accuracy": balanced_accuracy_score(y, hat), "Flagged share": hat.mean(),
                     "False positives": int((hat & (y == 0)).sum()), "Missed users": int((~hat & (y == 1)).sum())})
    return pd.DataFrame(rows)


def best_threshold(y, p):
    f1 = [f1_score(y, p >= t, zero_division=0) for t in GRID_T]
    return float(GRID_T[int(np.argmax(f1))])


def thresholds(train, valid, best, strategies):
    X, y = pl.xy(train)
    Xv, yv = pl.xy(valid)
    th, preds = {}, pd.DataFrame({"y": yv})
    for name in pl.MODEL_NAMES:
        p = pl.build(name, best[name], strategies[name]).fit(X, y).predict_proba(Xv)[:, 1]
        preds[name] = p
        th[name] = best_threshold(yv, p)
    save(preds, "validation_predictions")
    return th, preds


# 6 ------------------------------------------------------------------ final fit and single test evaluation
def metrics(y, p, t, w=None):
    hat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, hat, labels=[0, 1]).ravel()
    return {"Threshold": t, "Accuracy": accuracy_score(y, hat, sample_weight=w),
            "Balanced accuracy": balanced_accuracy_score(y, hat, sample_weight=w),
            "Precision": precision_score(y, hat, zero_division=0, sample_weight=w),
            "Recall (TPR)": recall_score(y, hat, sample_weight=w), "TNR": tn / (tn + fp) if w is None else np.nan,
            "F1": f1_score(y, hat, zero_division=0, sample_weight=w),
            "ROC-AUC": roc_auc_score(y, p, sample_weight=w), "AP (PR-AUC)": average_precision_score(y, p, sample_weight=w),
            "Brier": brier_score_loss(y, p, sample_weight=w), "Log loss": log_loss(y, p, sample_weight=w),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def final_fit(train, valid, best, strategies):
    """Refit every chosen pipeline on train + validation (80%)."""
    dev = pd.concat([train, valid])
    X, y = pl.xy(dev)
    fitted = {"Dummy": pl.build("Dummy").fit(X, y)}
    for name in pl.MODEL_NAMES:
        fitted[name] = pl.build(name, best[name], strategies[name]).fit(X, y)
    for name, m in fitted.items():
        joblib.dump(m, cfg.MODELS / f"{name.lower().replace(' ', '_')}.joblib", compress=3)
    return fitted


def test_evaluation(fitted, test, th, n_boot=1000):
    Xt, yt = pl.xy(test)
    w = test["weight"].to_numpy()
    preds = pd.DataFrame({"shedid": test["shedid"].to_numpy(), "y": yt, "weight": w})
    rows, wrows = [], []
    for name, m in fitted.items():
        p = m.predict_proba(Xt)[:, 1]
        preds[name] = p
        t = th.get(name, 0.5)
        rows.append({"Model": name, **metrics(yt, p, t)})
        wrows.append({"Model": name, **{k: v for k, v in metrics(yt, p, t, w).items() if k not in ("TP", "FP", "FN", "TN", "TNR")}})
        if name != "Dummy":
            rows[-1]["Recall @0.50"] = recall_score(yt, p >= .5)
            rows[-1]["F1 @0.50"] = f1_score(yt, p >= .5, zero_division=0)
    res = pd.DataFrame(rows)
    # Percentile bootstrap: resample test respondents, same resamples for every model (paired).
    rng = np.random.default_rng(cfg.SEED)
    idx = [rng.integers(0, len(yt), len(yt)) for _ in range(n_boot)]
    boot = {n: {"AP": np.array([average_precision_score(yt[i], preds[n].to_numpy()[i]) for i in idx]),
                "ROC": np.array([roc_auc_score(yt[i], preds[n].to_numpy()[i]) for i in idx])} for n in pl.MODEL_NAMES}
    ci = []
    ref = pl.MODEL_NAMES[0]
    for n in pl.MODEL_NAMES:
        d = boot[n]["AP"] - boot[ref]["AP"]
        ci.append({"Model": n, "AP low": np.percentile(boot[n]["AP"], 2.5), "AP high": np.percentile(boot[n]["AP"], 97.5),
                   "ROC low": np.percentile(boot[n]["ROC"], 2.5), "ROC high": np.percentile(boot[n]["ROC"], 97.5),
                   "AP diff vs LR": d.mean(), "diff low": np.percentile(d, 2.5), "diff high": np.percentile(d, 97.5)})
    save(preds, "test_predictions")
    save(res, "test_metrics")
    save(pd.DataFrame(wrows), "test_metrics_weighted")
    save(pd.DataFrame(ci), "test_bootstrap_ci")
    return res, pd.DataFrame(wrows), pd.DataFrame(ci), preds


def calibration(preds, bins=10):
    rows = []
    for n in pl.MODEL_NAMES:
        frac, mean = calibration_curve(preds["y"], preds[n], n_bins=bins, strategy="quantile")
        rows += [{"Model": n, "Mean predicted": a, "Observed rate": b} for a, b in zip(mean, frac)]
    return save(pd.DataFrame(rows), "calibration")


# 7 ------------------------------------------------------------------ interpretation
def coefficients(pipe):
    names, src = pl.transformed_names(pipe)
    coef = pipe.named_steps["clf"].coef_[0]
    t = pd.DataFrame({"Term": names, "Feature": src, "Coefficient": coef, "Odds ratio": np.exp(coef)})
    return save(t.reindex(t.Coefficient.abs().sort_values(ascending=False).index), "lr_coefficients")


def permutation(pipe, test):
    X, y = pl.xy(test)
    r = permutation_importance(pipe, X, y, scoring="average_precision", n_repeats=20, random_state=cfg.SEED, n_jobs=1)
    t = pd.DataFrame({"Feature": X.columns, "Label": [cfg.LABELS[c] for c in X.columns],
                      "Mean AP drop": r.importances_mean, "SD": r.importances_std})
    return save(t.sort_values("Mean AP drop", ascending=False), "permutation_importance")


def tree_importance(pipe):
    _, src = pl.transformed_names(pipe)
    t = pd.DataFrame({"Feature": src, "Importance": pipe.named_steps["clf"].feature_importances_})
    t = t.groupby("Feature", as_index=False)["Importance"].sum().sort_values("Importance", ascending=False)
    t.insert(1, "Label", t["Feature"].map(cfg.LABELS))
    return save(t, "rf_importance")


def shap_summary(pipe, background, sample):
    """SHAP on the fitted classifier, one-hot columns summed back to their source feature."""
    import shap
    prep = pipe[:-1]
    _, src = pl.transformed_names(pipe)
    B, S = prep.transform(background[cfg.FEATURES]), prep.transform(sample[cfg.FEATURES])
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        sv = shap.LinearExplainer(clf, shap.maskers.Independent(B, max_samples=len(B))).shap_values(S)
    else:
        sv = shap.TreeExplainer(clf).shap_values(S)
        sv = sv[..., 1] if np.ndim(sv) == 3 else sv
    agg = pd.DataFrame(sv, columns=src).T.groupby(level=0).sum().T
    t = pd.DataFrame({"Feature": agg.columns, "Mean |SHAP| (log-odds)": agg.abs().mean().to_numpy()})
    t.insert(1, "Label", t.Feature.map(cfg.LABELS))
    return save(t.sort_values("Mean |SHAP| (log-odds)", ascending=False), "shap_importance"), agg


# 8 ------------------------------------------------------------------ fairness audit and robustness
def add_groups(df):
    d = df.copy()
    d["Age group"] = pd.cut(d["age"], [17, 29, 44, 59, 120], labels=["18-29", "30-44", "45-59", "60+"]).astype(str)
    inc = d["income_band"].map({b: i for i, b in enumerate(cfg.INCOME_BANDS)})
    d["Income group"] = pd.cut(inc, [-1, 4, 8, 11, 14],
                               labels=["Under $25k", "$25k-$50k", "$50k-$100k", "$100k+"]).astype(str)
    return d


def fairness(test, preds, final, t):
    d = add_groups(test.reset_index(drop=True))
    p, y = preds[final].to_numpy(), preds["y"].to_numpy()
    rows = []
    for dim in ["gender", "race", "Age group", "Income group"]:
        for g in sorted(d[dim].unique()):
            m = (d[dim] == g).to_numpy()
            hat = p[m] >= t
            pos, neg = y[m] == 1, y[m] == 0
            rows.append({"Dimension": cfg.LABELS.get(dim, dim), "Group": g, "n": int(m.sum()), "BNPL users": int(pos.sum()),
                         "Actual rate": y[m].mean(), "Mean predicted": p[m].mean(), "Flagged share": hat.mean(),
                         "Recall (TPR)": hat[pos].mean() if pos.any() else np.nan,
                         "FPR": hat[neg].mean() if neg.any() else np.nan,
                         "Precision": y[m][hat].mean() if hat.any() else np.nan,
                         "ROC-AUC": roc_auc_score(y[m], p[m]) if pos.any() and neg.any() else np.nan})
    return save(pd.DataFrame(rows), "fairness_audit")


def learning(train, name, params, strategy):
    X, y = pl.xy(train)
    n, tr, te = learning_curve(pl.build(name, params, strategy), X, y, cv=CV, scoring="average_precision",
                               train_sizes=np.linspace(0.15, 1, 6), random_state=cfg.SEED, shuffle=True, n_jobs=1)
    return save(pd.DataFrame({"Training rows": n, "Train AP": tr.mean(1), "CV AP": te.mean(1),
                              "CV SE": te.std(1, ddof=1) / np.sqrt(te.shape[1])}), "learning_curve")


def weight_sensitivity(train, valid, name, params, strategy):
    """Does fitting with survey weights change the result? (validation set, weighted and unweighted AP)"""
    X, y = pl.xy(train)
    Xv, yv = pl.xy(valid)
    rows = []
    for label, kw in {"Unweighted fit": {}, "Survey-weighted fit": {"clf__sample_weight": train["weight"].to_numpy()}}.items():
        p = pl.build(name, params, strategy).fit(X, y, **kw).predict_proba(Xv)[:, 1]
        rows.append({"Fit": label, "Valid AP": average_precision_score(yv, p), "Valid ROC-AUC": roc_auc_score(yv, p),
                     "Weighted valid AP": average_precision_score(yv, p, sample_weight=valid["weight"]),
                     "Weighted valid ROC-AUC": roc_auc_score(yv, p, sample_weight=valid["weight"])})
    return save(pd.DataFrame(rows), "weight_sensitivity")


def save_metadata(final, strategies, th, best, splits):
    import sklearn
    meta = {"final_model": final, "strategy": strategies, "thresholds": th, "best_params": best, "seed": cfg.SEED,
            "features": cfg.FEATURES, "rows": {k: len(v) for k, v in splits.items()},
            "positive_rate": {k: float(v[cfg.TARGET].mean()) for k, v in splits.items()},
            "sklearn": sklearn.__version__}
    (cfg.MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta
