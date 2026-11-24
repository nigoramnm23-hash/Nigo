"""Static figures (matplotlib) for the notebook and report. Each function saves a PNG and returns the figure."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from . import config as cfg
from . import eda

C = cfg.C
EMERALD = LinearSegmentedColormap.from_list("emerald", ["#F2F7F4", "#0B6E4F"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#B4532A", "#F3F3F0", "#0B6E4F"])
LINE = {"solid": "-", "dash": "--", "dashdot": "-.", "dot": ":", "longdash": (0, (8, 3))}
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.family": "Arial", "font.size": 10, "axes.edgecolor": C["grid"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
    "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelcolor": C["muted"],
    "xtick.color": C["muted"], "ytick.color": C["muted"], "text.color": C["ink"], "legend.frameon": False,
    "text.parse_math": False})   # dollar signs in income labels are text, not formulas


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    return fig


def target(df):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    n = df[cfg.TARGET].value_counts()
    w = np.average(df[cfg.TARGET], weights=df["weight"])
    ax.barh(["Did not use BNPL", "Used BNPL"], [n[0], n[1]], color=[C["non"], C["user"]], height=.55)
    for i, v in enumerate([n[0], n[1]]):
        ax.text(v + 120, i, f"{v:,} ({v / len(df):.1%})", va="center")
    ax.set_xlim(0, n[0] * 1.22); ax.grid(axis="y", visible=False)
    ax.set_title(f"Target distribution, n = {len(df):,} (survey-weighted BNPL rate {w:.1%})")
    ax.set_xlabel("Respondents")
    return _save(fig, "01_target")


def missing(raw):
    m = raw.drop(columns=["shedid", "weight", "BNPL1", "BNPL1_iflag"]).isna().mean().mul(100)
    m = m[m > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.barh(m.index, m, color=C["user"], height=.55)
    for i, v in enumerate(m):
        ax.text(v + .4, i, f"{v:.1f}%", va="center")
    ax.grid(axis="y", visible=False); ax.set_xlim(0, m.max() * 1.15)
    ax.set_title("Raw variables with blanks: C4A and EF5D are skip-logic, ppfs1482 is true non-response")
    ax.title.set_fontsize(9); ax.set_xlabel("Blank (%)")
    return _save(fig, "02_missing")


def histograms(df):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for ax, c in zip(axes.ravel(), cfg.NUMERIC):
        bins = np.arange(18, 103, 5) if c == "age" else np.arange(-.5, df[c].max() + 1.5)
        for v, col, lab in [(0, C["non"], "Did not use BNPL"), (1, C["user"], "Used BNPL")]:
            ax.hist(df.loc[df[cfg.TARGET] == v, c], bins=bins, density=True, histtype="stepfilled" if v == 0 else "step",
                    color=col, lw=2, label=lab)
        ax.set_title(cfg.LABELS[c], fontsize=9.5); ax.set_ylabel("Density")
    axes.ravel()[-1].axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l, loc="center")
    return _save(fig, "03_histograms")


def boxplots(df):
    cols = ["age", "household_size", "medical_skips", "afs_count"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.4))
    for ax, c in zip(axes, cols):
        b = ax.boxplot([df.loc[df[cfg.TARGET] == v, c] for v in (0, 1)], widths=.55, patch_artist=True,
                       tick_labels=["Non-user", "User"], flierprops=dict(marker="o", ms=3, alpha=.25, mec=C["muted"]),
                       medianprops=dict(color=C["ink"], lw=2))
        for patch, col in zip(b["boxes"], [C["non"], C["user"]]):
            patch.set(facecolor=col, alpha=.75, edgecolor=C["ink"])
        ax.set_title(cfg.LABELS[c].split(" (")[0], fontsize=9.5)
    return _save(fig, "04_boxplots")


def rates(df, by, order, title, name, ax=None, short=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.6, .42 * len(order) + 1.2))
    t = eda.rate_table(df, by, order)
    y = np.arange(len(t))
    ax.barh(y, t["Weighted rate"] * 100, color=C["user"], height=.6)
    ax.errorbar(t["Unweighted rate"] * 100, y, xerr=[(t["Unweighted rate"] - t["CI low"]) * 100, (t["CI high"] - t["Unweighted rate"]) * 100],
                fmt="o", ms=4, color=C["ink"], lw=1, capsize=2)
    labs = [(short or {}).get(v, str(v)) for v in t[by]]
    ax.set_yticks(y, [f"{l} (n={n:,})" for l, n in zip(labs, t["n"])], fontsize=8.5)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("BNPL use (%)")
    ax.set_title(title, fontsize=10)
    return _save(fig, name) if own else ax


NOTE = "Bars: survey-weighted rate. Dots: unweighted sample rate with 95% Wilson interval."


def demographics(df):
    from .experiments import add_groups
    d = add_groups(df)
    specs = [("Age group", ["18-29", "30-44", "45-59", "60+"], "Age"), ("education", cfg.EDUCATION, "Education"),
             ("partner_status", list(d.partner_status.value_counts().index), "Partnership"),
             ("housing", list(d.housing.value_counts().index), "Housing"),
             ("gender", ["Female", "Male"], "Gender (audit only)"),
             ("race", ["White", "Black", "Hispanic", "Asian", "Other"], "Race/ethnicity (audit only)")]
    short = {"Less than a high school degree": "< High school", "High school degree or GED": "High school",
             "Some college/technical or associates degree": "Some college", "Bachelor's degree or more": "Bachelor's+",
             "Divorced, separated or widowed": "Previously married"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), sharex=True)
    for ax, (b, o, t) in zip(axes.ravel(), specs):
        rates(d, b, o, t, None, ax, short)
    fig.text(0.01, -0.01, NOTE, fontsize=8.5, color=C["muted"])
    return _save(fig, "05_demographics")


def income(df):
    f = rates(df, "income_band", cfg.INCOME_BANDS, "BNPL use by annual income (respondent and partner)", "06_income")
    return f


def finance(df):
    specs = [("financial_wellbeing", cfg.WELLBEING, "Financial well-being"),
             ("bills_status", ["Not all paid in full", "Paid in full with difficulty", "Paid in full without difficulty"], "Bills last month"),
             ("card_behaviour", ["No credit card", "Pays in full", "Carried balance once", "Carries balance sometimes", "Carries balance mostly"], "Credit-card behaviour"),
             ("credit_score", ["Very poor", "Poor", "Fair", "Good", "Excellent", "Don't know"], "Self-rated credit score"),
             ("emergency_fund", cfg.YESNO, "Emergency fund"), ("can_cover_400", cfg.YESNO, "Can cover $400")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.2), sharex=True)
    for ax, (b, o, t) in zip(axes.ravel(), specs):
        rates(df, b, o, t, None, ax)
    fig.text(0.01, -0.01, NOTE, fontsize=8.5, color=C["muted"])
    return _save(fig, "07_finance")


def heat_income_wellbeing(df):
    from .experiments import add_groups
    d = add_groups(df)
    cols = ["Under $25k", "$25k-$50k", "$50k-$100k", "$100k+"]
    r = d.pivot_table(index="financial_wellbeing", columns="Income group", values=cfg.TARGET, aggfunc=["mean", "size"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    m = r["mean"].loc[cfg.WELLBEING, cols].to_numpy() * 100
    n = r["size"].loc[cfg.WELLBEING, cols].to_numpy()
    ax.imshow(m, cmap=EMERALD, aspect="auto"); ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{m[i, j]:.0f}%\nn={n[i, j]:,}", ha="center", va="center", fontsize=8.5,
                    color="white" if m[i, j] > m.max() * .6 else C["ink"])
    ax.set_xticks(range(4), cols); ax.set_yticks(range(4), cfg.WELLBEING)
    ax.set_title("Unweighted BNPL use by income and financial well-being", fontsize=10)
    return _save(fig, "08_income_wellbeing")


def age_by_wellbeing(df):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = ["#0B3D2E", "#0B6E4F", "#4FA585", "#A9B4AE"]
    bands = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    cut = np.digitize(df.age, [30, 40, 50, 60, 70])
    for w, c, st in zip(cfg.WELLBEING, cols, ["-", "--", "-.", ":"]):
        g = df[df.financial_wellbeing == w].groupby(cut[df.financial_wellbeing == w])[cfg.TARGET].mean()
        ax.plot(range(6), g * 100, marker="o", ms=6, lw=2, ls=st, color=c, label=w)
    ax.set_xticks(range(6), bands); ax.set_xlabel("Age"); ax.set_ylabel("BNPL use (%)"); ax.set_ylim(0)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("BNPL use by age within each level of financial well-being (unweighted)", fontsize=10)
    return _save(fig, "09_age_wellbeing")


def spearman(df):
    m = eda.spearman_matrix(df)
    labs = [cfg.LABELS.get(c, c).split(" (")[0] for c in m.columns]
    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    ax.imshow(m, cmap=DIVERGE, vmin=-1, vmax=1); ax.grid(False)
    for i in range(len(m)):
        for j in range(len(m)):
            v = m.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="white" if abs(v) > .6 else C["ink"])
    ax.set_xticks(range(len(m)), labs, rotation=90, fontsize=7.5); ax.set_yticks(range(len(m)), labs, fontsize=7.5)
    ax.set_title("Spearman rank correlation (numeric and ordinal variables)", fontsize=10.5)
    return _save(fig, "10_spearman")


def association(t, top=20):
    d = t.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    col = [C["user"] if m.startswith("Cramer") else C["accent"] for m in d.Measure]
    ax.barh(d.Label, d.Strength.abs(), color=col, height=.62)
    ax.grid(axis="y", visible=False); ax.set_xlabel("Absolute strength of association with BNPL use")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["user"]), plt.Rectangle((0, 0), 1, 1, color=C["accent"])],
              labels=["Cramer's V (categorical/ordinal)", "|Rank-biserial| (numeric)"], loc="lower right", fontsize=8.5)
    ax.set_title("Strongest single-variable associations (the two measures are not on one scale)", fontsize=9.5)
    return _save(fig, "11_association")


def _curve_axes(preds, kind, models):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    y = preds["y"]
    for n in models:
        if kind == "roc":
            a, b, _ = roc_curve(y, preds[n]); lab = f"{n} (AUC {roc_auc_score(y, preds[n]):.3f})"
        else:
            b, a, _ = precision_recall_curve(y, preds[n]); lab = f"{n} (AP {average_precision_score(y, preds[n]):.3f})"
        ax.plot(a, b, color=cfg.MODEL_COLOURS[n], ls=LINE[cfg.MODEL_DASH[n]], lw=2, label=lab)
    return fig, ax


def roc(preds, models):
    fig, ax = _curve_axes(preds, "roc", models)
    ax.plot([0, 1], [0, 1], color=C["non"], lw=1); ax.legend(fontsize=8.5, loc="lower right")
    ax.set(xlabel="False positive rate", ylabel="True positive rate"); ax.set_title("ROC curves, test set")
    return _save(fig, "20_roc")


def pr(preds, models):
    fig, ax = _curve_axes(preds, "pr", models)
    base = preds["y"].mean()
    ax.axhline(base, color=C["non"], lw=1); ax.text(.62, base - .035, f"no-skill = {base:.3f}", fontsize=8.5, color=C["muted"])
    ax.legend(fontsize=8.5); ax.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1)); ax.set_title("Precision-recall curves, test set")
    return _save(fig, "21_pr")


def confusion(preds, th, models):
    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 3.2))
    for ax, n in zip(axes, models):
        cm = confusion_matrix(preds["y"], preds[n] >= th[n])
        share = cm / cm.sum(1, keepdims=True)
        ax.imshow(share, cmap=EMERALD, vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center", color="white" if share[i, j] > .6 else C["ink"])
        ax.set_xticks([0, 1], ["Pred. non-user", "Pred. user"], fontsize=8); ax.set_yticks([0, 1], ["Non-user", "User"], fontsize=8)
        ax.set_title(f"{n}\n(threshold {th[n]:.2f})", fontsize=9)
    return _save(fig, "22_confusion")


def threshold(tab, t):
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for c, col, st in [("Precision", "#2A78D6", "--"), ("Recall", "#EB6834", "-."), ("F1", C["user"], "-")]:
        ax.plot(tab.Threshold, tab[c], color=col, ls=st, lw=2, label=c)
    ax.axvline(t, color=C["ink"], lw=1); ax.text(t + .01, .95, f"chosen {t:.2f}", fontsize=8.5)
    ax.axvline(.5, color=C["non"], lw=1); ax.text(.51, .95, "default 0.50", fontsize=8.5, color=C["muted"])
    ax.set(xlabel="Decision threshold", ylabel="Score", ylim=(0, 1)); ax.legend(loc="center right")
    ax.set_title("Threshold trade-off (validation set, final model)")
    return _save(fig, "23_threshold")


def calibration(cal, models):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, .7], [0, .7], color=C["non"], lw=1)
    for n in models:
        d = cal[cal.Model == n]
        ax.plot(d["Mean predicted"], d["Observed rate"], color=cfg.MODEL_COLOURS[n], ls=LINE[cfg.MODEL_DASH[n]], marker="o", ms=5, lw=1.8, label=n)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed BNPL rate"); ax.legend(fontsize=8.5)
    ax.set_title("Calibration by decile, test set")
    return _save(fig, "24_calibration")


def hbar(t, label, value, title, name, top=15, err=None, xlabel=None):
    d = t.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, .33 * len(d) + 1.2))
    ax.barh(d[label], d[value], color=C["user"], height=.62, xerr=d[err] if err else None,
            error_kw=dict(ecolor=C["ink"], lw=1))
    ax.grid(axis="y", visible=False); ax.set_xlabel(xlabel or value); ax.set_title(title, fontsize=10)
    return _save(fig, name)


def coefs(t, top=20):
    d = t.head(top).iloc[::-1]
    lab = [f"{cfg.LABELS[f]}" + ("" if term == f else f": {term[len(f) + 1:]}") for f, term in zip(d.Feature, d.Term)]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.barh(lab, d.Coefficient, color=[C["user"] if v > 0 else C["accent"] for v in d.Coefficient], height=.62)
    ax.axvline(0, color=C["ink"], lw=1); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Coefficient in log-odds (numeric and ordinal inputs are standardised)")
    ax.set_title("Logistic Regression: largest coefficients (right = higher propensity)", fontsize=10)
    return _save(fig, "25_coefficients")


def imbalance(imb):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    strat = list(imb.Strategy.unique())
    cols = [C["user"], "#2A78D6", "#EB6834", "#6250D6"]
    models = list(imb.Model.unique())
    for ax, (m, t) in zip(axes, [("CV AP", "CV average precision (higher better)"), ("CV Brier", "CV Brier score (lower better)"),
                                 ("Recall @0.50", "Validation recall at 0.50")]):
        for k, s in enumerate(strat):
            d = imb[imb.Strategy == s].set_index("Model").loc[models]
            ax.barh(np.arange(len(models)) + (k - 1.5) * .2, d[m], height=.18, color=cols[k], label=s)
        ax.set_yticks(range(len(models)), models); ax.set_title(t, fontsize=9.5); ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.06))
    return _save(fig, "26_imbalance")


def fairness(t):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    lab = t.Dimension + ": " + t.Group
    for ax, (a, b, ttl) in zip(axes, [("Actual rate", "Mean predicted", "Calibration by group"),
                                      ("Recall (TPR)", "FPR", "Error rates by group at the chosen threshold")]):
        y = np.arange(len(t))
        ax.hlines(y, t[a], t[b], color=C["non"], lw=2)
        ax.plot(t[a], y, "o", ms=8, color=C["user"], label=a); ax.plot(t[b], y, "D", ms=7, color=C["accent"], label=b)
        ax.set_yticks(y, lab, fontsize=8.5); ax.set_title(ttl, fontsize=10); ax.legend(fontsize=8.5)
    axes[0].invert_yaxis()
    return _save(fig, "27_fairness")


def learning(t):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(t["Training rows"], t["Train AP"], color=C["accent"], ls="--", marker="o", lw=2, label="Training folds")
    ax.plot(t["Training rows"], t["CV AP"], color=C["user"], marker="o", lw=2, label="Held-out folds")
    ax.fill_between(t["Training rows"], t["CV AP"] - t["CV SE"], t["CV AP"] + t["CV SE"], color=C["user"], alpha=.15)
    ax.set(xlabel="Training rows", ylabel="Average precision"); ax.legend(); ax.set_title("Learning curve, final model")
    return _save(fig, "28_learning_curve")


def shap_strip(agg, X, imp, top=12):
    """SHAP dot plot: each dot is a test respondent; colour = feature rank (low to high) where ordered."""
    fig, ax = plt.subplots(figsize=(7.6, 5))
    rng = np.random.default_rng(0)
    ranks = eda.codes(X).rank(pct=True)
    feats = list(imp.Feature[:top])[::-1]
    for i, f in enumerate(feats):
        v = agg[f].to_numpy()
        colr = ranks[f].to_numpy() if f in ranks else None
        ax.scatter(v, i + rng.uniform(-.28, .28, len(v)), s=5, c=colr if colr is not None else C["non"],
                   cmap=EMERALD if colr is not None else None, alpha=.65, lw=0)
    ax.set_yticks(range(len(feats)), [cfg.LABELS[f] for f in feats], fontsize=8.5); ax.axvline(0, color=C["ink"], lw=1)
    ax.set_xlabel("SHAP value (log-odds). Dark = high value, light = low; grey = nominal or derived feature")
    ax.set_title("What moves individual predictions (test set)", fontsize=10)
    return _save(fig, "29_shap")
