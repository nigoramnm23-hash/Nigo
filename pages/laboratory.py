import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

from bnpl import config as cfg, experiments as ex, pipeline as pl, ui

ui.page("ML laboratory", "Train a model yourself on the training split and judge it on the validation split. The test set stays locked away, "
        "so experimenting here cannot leak into the reported results.")
train, valid, _ = ui.splits()
HELP = {
    "Logistic Regression": "Adds up evidence from each characteristic. Transparent and usually well calibrated; assumes effects add on the log-odds scale.",
    "Decision Tree": "A sequence of yes/no splits. Easy to read; unstable and prone to overfitting when deep.",
    "Random Forest": "Averages many varied trees. Captures interactions; less transparent.",
    "Gradient Boosting": "Builds small trees one after another, each correcting the last. Strong on tables; needs careful tuning.",
}


@st.cache_data(show_spinner=False, max_entries=40)
def run(name, params, strategy, n):
    tr = train if n >= len(train) else train.sample(n, random_state=cfg.SEED)
    X, y = pl.xy(tr)
    Xv, yv = pl.xy(valid)
    t0 = time.time()
    pipe = pl.build(name, dict(params), strategy).fit(X, y)
    return pipe.predict_proba(X)[:, 1], y, pipe.predict_proba(Xv)[:, 1], yv, time.time() - t0


name = st.segmented_control("Algorithm", pl.MODEL_NAMES, default=pl.MODEL_NAMES[0], required=True)
st.caption(HELP[name])
with st.form("lab"):
    a, b, c = st.columns(3)
    p = {}
    if name == "Logistic Regression":
        p["C"] = a.select_slider("C (smaller = stronger regularisation)", [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0], 0.1)
        p["l1_ratio"] = b.select_slider("L1 share (0 = ridge, 1 = lasso)", [0.0, 0.5, 1.0], 0.0)
    elif name == "Decision Tree":
        p["max_depth"] = a.slider("Maximum depth", 1, 20, 6)
        p["min_samples_leaf"] = b.slider("Minimum rows per leaf", 1, 200, 50)
        p["criterion"] = c.selectbox("Split criterion", ["gini", "log_loss"])
    elif name == "Random Forest":
        p["n_estimators"] = a.slider("Trees (limit 300)", 50, 300, 150, 50)
        p["max_depth"] = b.slider("Maximum depth", 2, 20, 10)
        p["min_samples_leaf"] = c.slider("Minimum rows per leaf", 1, 50, 5)
    else:
        p["learning_rate"] = a.select_slider("Learning rate", [0.02, 0.05, 0.1, 0.2], 0.05)
        p["max_iter"] = b.slider("Boosting rounds (limit 300)", 50, 300, 150, 50)
        p["max_leaf_nodes"] = c.slider("Leaves per tree", 3, 31, 8)
    a, b = st.columns(2)
    strategy = a.selectbox("Class-balancing strategy", pl.STRATEGIES)
    n = b.slider("Training rows", 1000, len(train), len(train), 500, help="Use fewer rows for a faster run.")
    go_btn = st.form_submit_button("Train model", type="primary", icon=":material/play_arrow:")

if go_btn:
    with st.spinner(f"Training {name} on {n:,} rows"):
        st.session_state["lab_result"] = (name, p, strategy, n, run(name, tuple(sorted(p.items())), strategy, n))
        r = st.session_state["lab_result"][4]
        st.session_state.setdefault("lab_log", []).append({
            "Model": name, "Strategy": strategy, "Rows": n, "Settings": ", ".join(f"{k}={v}" for k, v in p.items()),
            "Train AP": average_precision_score(r[1], r[0]), "Validation AP": average_precision_score(r[3], r[2])})

if "lab_result" not in st.session_state:
    st.info("Choose settings and select Train model. A run usually takes between one and thirty seconds.")
    st.stop()

name_r, p_r, strat_r, n_r, (ptr, ytr, pv, yv, secs) = st.session_state["lab_result"]
t = st.slider("Decision threshold", 0.02, 0.90, ex.best_threshold(yv, pv), 0.01, help="Starts at the value with the best validation F1.")
mt, mv = ex.metrics(ytr, ptr, t), ex.metrics(yv, pv, t)
gap = mt["AP (PR-AUC)"] - mv["AP (PR-AUC)"]
ui.kpis([("Validation average precision", f"{mv['AP (PR-AUC)']:.3f}", f"no-skill {yv.mean():.3f}", True),
         ("Training average precision", f"{mt['AP (PR-AUC)']:.3f}", f"gap {gap:+.3f}" + (" (overfitting)" if gap > .08 else "")),
         ("Validation ROC-AUC", f"{mv['ROC-AUC']:.3f}", f"Brier {mv['Brier']:.3f}"),
         ("Recall / precision", f"{mv['Recall (TPR)']:.0%} / {mv['Precision']:.0%}", f"at threshold {t:.2f}"),
         ("Run time", f"{secs:.1f} s", f"{name_r}, {strat_r.lower()}")])
a, b = st.columns(2)
with a:
    keep = ["Accuracy", "Balanced accuracy", "Precision", "Recall (TPR)", "F1", "ROC-AUC", "AP (PR-AUC)", "Brier", "Log loss"]
    st.dataframe(pd.DataFrame({"Metric": keep, "Training": [mt[k] for k in keep], "Validation": [mv[k] for k in keep]}).round(3),
                 hide_index=True, width="stretch")
    fig = go.Figure(go.Heatmap(z=[[mv["TN"], mv["FP"]], [mv["FN"], mv["TP"]]], x=["Predicted non-user", "Predicted user"],
                               y=["Non-user", "User"], colorscale=ui.EMERALD, showscale=False, texttemplate="%{z:,}"))
    fig.update_yaxes(autorange="reversed"); fig.update_layout(title="Validation confusion matrix"); ui.chart(fig, 300)
with b:
    fig = go.Figure()
    for yy, pp, lab, col, dash in [(ytr, ptr, "Training", cfg.C["accent"], "dash"), (yv, pv, "Validation", cfg.C["user"], "solid")]:
        pr, rc, _ = precision_recall_curve(yy, pp)
        s = max(1, len(pr) // 600)
        fig.add_scatter(x=rc[::s], y=pr[::s], name=lab, line=dict(color=col, dash=dash, width=2))
    fig.add_hline(y=yv.mean(), line_color=cfg.C["non"]); fig.update_layout(title="Precision-recall")
    fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision", range=[0, 1]); ui.chart(fig, 380)
if st.session_state.get("lab_log"):
    st.markdown("**Your runs this session**")
    st.dataframe(pd.DataFrame(st.session_state["lab_log"]).round(3), hide_index=True, width="stretch")
