import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from bnpl import config as cfg, experiments as ex, pipeline as pl, ui

ui.page("Model performance", "How four tuned models and a baseline perform on the 2,581 respondents that were kept aside until the end.")
m, res, preds = ui.meta(), ui.table("test_metrics"), ui.table("test_predictions")
final, th = m["final_model"], m["thresholds"]
r = res.set_index("Model").loc[final]
ci = ui.table("test_bootstrap_ci").set_index("Model").loc[final]
ui.kpis([("Selected model", final, f"threshold {th[final]:.2f}", True),
         ("Average precision", f"{r['AP (PR-AUC)']:.3f}", f"95% interval {ci['AP low']:.3f} to {ci['AP high']:.3f}"),
         ("ROC-AUC", f"{r['ROC-AUC']:.3f}", f"95% interval {ci['ROC low']:.3f} to {ci['ROC high']:.3f}"),
         ("Recall / precision", f"{r['Recall (TPR)']:.0%} / {r['Precision']:.0%}", "share of users found / share of flags correct"),
         ("Brier score", f"{r['Brier']:.3f}", "probability error, lower is better")])
t1, t2, t3, t4, t5 = st.tabs(["Comparison", "Curves", "Thresholds", "Explanation", "Fairness and robustness"])
with t1:
    basis = st.segmented_control("Estimates", ["Unweighted test sample", "Survey-weighted"], default="Unweighted test sample", required=True)
    show = res.drop(columns=["TP", "FP", "FN", "TN"]) if basis.startswith("Un") else ui.table("test_metrics_weighted")
    st.dataframe(ui.fmt(show), hide_index=True, width="stretch")
    metric = st.selectbox("Compare models on", ["AP (PR-AUC)", "ROC-AUC", "F1", "Recall (TPR)", "Precision", "Balanced accuracy", "Accuracy", "Brier"])
    fig = go.Figure(go.Bar(x=show[metric], y=show["Model"], orientation="h", text=show[metric].round(3), textposition="outside", cliponaxis=False,
                           marker_color=[cfg.MODEL_COLOURS[x] for x in show["Model"]], hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title=f"{metric}, test set"); fig.update_yaxes(autorange="reversed"); ui.chart(fig, 320)
    if metric == "Accuracy":
        st.info("The Dummy model has high accuracy because most people are non-users, yet it finds no BNPL users at all.")
    st.markdown("**Why this model was selected** (rule fixed before results were seen: the simplest model within one standard error of the best cross-validated score)")
    st.dataframe(ui.fmt(ui.table("model_selection"), dec=4), hide_index=True, width="stretch")
    st.markdown("**Tuning and overfitting**")
    st.dataframe(ui.fmt(ui.table("tuning_summary"), dec=4), hide_index=True, width="stretch")
with t2:
    pick = st.multiselect("Models", pl.MODEL_NAMES, default=pl.MODEL_NAMES)
    a, b = st.columns(2)
    y = preds["y"]
    f1, f2 = go.Figure(), go.Figure()
    for n in pick:
        st_ = dict(color=cfg.MODEL_COLOURS[n], dash=cfg.MODEL_DASH[n], width=2)
        fp, tp, _ = roc_curve(y, preds[n]); f1.add_scatter(x=fp, y=tp, name=f"{n} ({roc_auc_score(y, preds[n]):.3f})", line=st_)
        p, q, _ = precision_recall_curve(y, preds[n]); f2.add_scatter(x=q, y=p, name=f"{n} ({average_precision_score(y, preds[n]):.3f})", line=st_)
    f1.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=cfg.C["non"], width=1), showlegend=False, hoverinfo="skip")
    f2.add_hline(y=y.mean(), line_color=cfg.C["non"], annotation_text="no skill")
    f1.update_layout(title="ROC curves (AUC)"); f1.update_xaxes(title="False positive rate"); f1.update_yaxes(title="True positive rate")
    f2.update_layout(title="Precision-recall curves (AP)"); f2.update_xaxes(title="Recall"); f2.update_yaxes(title="Precision", range=[0, 1])
    with a: ui.chart(f1, 430, "roc")
    with b: ui.chart(f2, 430, "pr")
    cal = ui.table("calibration")
    fig = go.Figure(go.Scatter(x=[0, .7], y=[0, .7], line=dict(color=cfg.C["non"], width=1), showlegend=False, hoverinfo="skip"))
    for n in pick:
        d = cal[cal.Model == n]
        fig.add_scatter(x=d["Mean predicted"], y=d["Observed rate"], mode="lines+markers", name=n, line=dict(color=cfg.MODEL_COLOURS[n], dash=cfg.MODEL_DASH[n], width=2), marker_size=8)
    fig.update_layout(title="Calibration by decile: on the grey line, predicted equals observed"); fig.update_xaxes(title="Mean predicted probability"); fig.update_yaxes(title="Observed BNPL rate")
    ui.chart(fig, 420, "cal")
with t3:
    tab = ui.table("threshold_table")
    fig = go.Figure()
    for c, col, dash in [("Precision", "#2A78D6", "dash"), ("Recall", "#EB6834", "dashdot"), ("F1", cfg.C["user"], "solid")]:
        fig.add_scatter(x=tab.Threshold, y=tab[c], name=c, line=dict(color=col, dash=dash, width=2))
    fig.add_vline(x=th[final], line_color=cfg.C["ink"], annotation_text=f"chosen {th[final]:.2f}")
    fig.update_layout(title="Threshold trade-off on validation data (where the threshold was chosen)"); fig.update_xaxes(title="Threshold"); fig.update_yaxes(title="Score")
    ui.chart(fig, 380)
    t = st.slider("Explore a threshold on the test set", 0.02, 0.90, float(th[final]), 0.01, help="For understanding only. The reported threshold was chosen on validation data.")
    mm = ex.metrics(preds["y"].to_numpy(), preds[final].to_numpy(), t)
    ui.kpis([("BNPL users found", f"{mm['TP']} of {mm['TP'] + mm['FN']}", f"recall {mm['Recall (TPR)']:.0%}"), ("Users missed", mm["FN"], "false negatives"),
             ("False alarms", mm["FP"], "non-users flagged"), ("Precision", f"{mm['Precision']:.0%}", "flags that are correct"), ("Accuracy", f"{mm['Accuracy']:.1%}", "can mislead here")])
    a, b = st.columns(2)
    for holder, n in zip([a, b] * 2, pl.MODEL_NAMES):
        x = ex.metrics(preds["y"].to_numpy(), preds[n].to_numpy(), t if n == final else th[n])
        fig = go.Figure(go.Heatmap(z=[[x["TN"], x["FP"]], [x["FN"], x["TP"]]], x=["Pred. non-user", "Pred. user"], y=["Non-user", "User"], colorscale=ui.EMERALD, showscale=False, texttemplate="%{z:,}"))
        fig.update_yaxes(autorange="reversed"); fig.update_layout(title=f"{n} (threshold {x['Threshold']:.2f})")
        with holder: ui.chart(fig, 280, f"cm{n}")
with t4:
    what = st.selectbox("Importance measure", ["Permutation importance (final model, test set)", "Mean absolute SHAP (final model)", "Logistic Regression coefficients", "Random Forest impurity importance"])
    if what.startswith("Perm"):
        d, x, lab = ui.table("permutation_importance").head(15), "Mean AP drop", "Label"
        st.caption("How much average precision falls when one input is shuffled. Correlated inputs share credit, so each can look smaller than it is.")
    elif what.startswith("Mean"):
        d, x, lab = ui.table("shap_importance").head(15), "Mean |SHAP| (log-odds)", "Label"
        st.caption("Average size of each input's contribution to a prediction, in log-odds.")
    elif what.startswith("Log"):
        d, x, lab = ui.table("lr_coefficients").head(20).copy(), "Coefficient", "Label"
        d["Label"] = [cfg.LABELS[f] + ("" if f == tm else f": {tm[len(f) + 1:]}") for f, tm in zip(d.Feature, d.Term)]
        st.caption("Positive = higher propensity, holding the other inputs fixed. Numeric and ordinal inputs are standardised.")
    else:
        d, x, lab = ui.table("rf_importance").head(15), "Importance", "Label"
        st.caption("Impurity importance favours inputs with many values, such as age. Compare it with permutation importance.")
    fig = go.Figure(go.Bar(x=d[x], y=d[lab], orientation="h", marker_color=[cfg.C["user"] if v >= 0 else cfg.C["accent"] for v in d[x]], hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); fig.update_layout(title=what); ui.chart(fig, 520)
    ui.limits("These are predictive associations, not causes. A characteristic can be a reason for using BNPL, a result of it, or a marker of something unmeasured.")
with t5:
    st.markdown("Gender and race are not model inputs. They are used here only to check whether the model works equally well for different groups.")
    f = ui.table("fairness_audit")
    dim = st.segmented_control("Group by", list(f.Dimension.unique()), default=f.Dimension.unique()[0], required=True)
    d = f[f.Dimension == dim]
    fig = go.Figure()
    for c, col, sy in [("Actual rate", cfg.C["user"], "circle"), ("Mean predicted", cfg.C["accent"], "diamond"), ("Recall (TPR)", "#2A78D6", "square"), ("FPR", "#6250D6", "x")]:
        fig.add_scatter(x=d[c] * 100, y=d.Group, mode="markers", name=c, marker=dict(size=11, color=col, symbol=sy))
    fig.update_layout(title=f"{dim}: rates (%), test set"); fig.update_xaxes(title="%"); ui.chart(fig, 340)
    st.dataframe(ui.fmt(d, pct=("Actual rate", "Mean predicted", "Flagged share", "Recall (TPR)", "FPR", "Precision")), hide_index=True, width="stretch")
    st.caption("Small groups have wide uncertainty; check the number of BNPL users before comparing.")
    a, b = st.columns(2)
    with a:
        lc = ui.table("learning_curve")
        fig = go.Figure([go.Scatter(x=lc["Training rows"], y=lc["Train AP"], name="Training folds", line=dict(color=cfg.C["accent"], dash="dash")),
                         go.Scatter(x=lc["Training rows"], y=lc["CV AP"], name="Held-out folds", line=dict(color=cfg.C["user"]))])
        fig.update_layout(title="Learning curve (average precision)"); ui.chart(fig, 320)
    with b:
        st.markdown("**Does fitting with survey weights change the result?** (validation set)")
        st.dataframe(ui.fmt(ui.table("weight_sensitivity"), dec=4), hide_index=True, width="stretch")
