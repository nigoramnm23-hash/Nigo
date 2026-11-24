import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bnpl import config as cfg, pipeline as pl, ui

ui.page("Data preparation", "From 815 raw survey columns to 26 documented model inputs, and how each preprocessing choice changes the data.")
df = ui.clean()
train, valid, test = ui.splits()
ui.kpis([("Raw file", "12,934 x 815", "public2025.csv, unmodified"), ("After cleaning", f"{len(df):,} rows", "31 imputed target answers dropped"),
         ("Model inputs", len(cfg.FEATURES), f"plus {len(cfg.DERIVED)} engineered in the pipeline"),
         ("Train / validation / test", f"{len(train):,} / {len(valid):,} / {len(test):,}", "stratified 60 / 20 / 20", True)])

t1, t2, t3, t4, t5 = st.tabs(["Cleaning", "Features", "Missing values", "Try preprocessing", "Class imbalance"])
with t1:
    st.markdown("Every correction is logged by the cleaning code. The rule: when the older panel profile and the live October 2025 "
                "survey answer disagree, the live answer wins.")
    st.dataframe(ui.table("cleaning_log"), hide_index=True, width="stretch")
    st.markdown("**Before and after: skip-logic blanks become information.** Credit-card balance (C4A) is blank only for people without a card, "
                "so the blank becomes its own category instead of being imputed.")
    c = df["card_behaviour"].value_counts()
    before = c.drop("No credit card"); before["(blank)"] = c["No credit card"]
    fig = go.Figure([go.Bar(name="Before (C4A)", y=before.index, x=before, orientation="h", marker_color=cfg.C["non"], text=before, textposition="outside"),
                     go.Bar(name="After (card_behaviour)", y=c.index, x=c, orientation="h", marker_color=cfg.C["user"], text=c, textposition="outside")])
    fig.update_layout(title="Respondents by category", barmode="group"); ui.chart(fig, 380)
with t2:
    fd = ui.table("feature_dictionary")
    q = st.text_input("Search the feature dictionary", placeholder="for example: credit, income, savings")
    show = fd if not q else fd[fd.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(show, hide_index=True, width="stretch")
    st.markdown("**Excluded variables and why**")
    st.dataframe(ui.table("excluded_variables"), hide_index=True, width="stretch")
    st.markdown("**Engineered inside the pipeline**: equivalised income (band midpoint divided by the square root of household size, logged), "
                "share of children in the household, and a 0-6 financial fragility score (struggling, no emergency fund, cannot cover \\$400, "
                "bill difficulty, overspending, skipped care). The top income band is open-ended; \\$250,000 is an assumption.")
with t3:
    miss = df[cfg.FEATURES].isna().mean().mul(100).round(2)
    miss = miss[miss > 0].rename("Missing after cleaning (%)").reset_index().rename(columns={"index": "Feature"})
    miss["Feature"] = miss["Feature"].map(cfg.LABELS)
    st.dataframe(miss, hide_index=True, width="stretch")
    st.markdown("Self-rated credit score is the only input with real non-response. Two treatments were compared by cross-validation on training data:")
    st.dataframe(ui.fmt(ui.table("preprocessing_choices"), dec=4), hide_index=True, width="stretch")
    rate = df.assign(cs=df.credit_score.fillna("Not reported")).groupby("cs")[cfg.TARGET].agg(["mean", "size"]).sort_values("mean")
    fig = go.Figure(go.Bar(y=rate.index, x=rate["mean"] * 100, orientation="h", marker_color=cfg.C["user"], text=[f"{a:.1f}% (n={b:,})" for a, b in zip(rate["mean"] * 100, rate["size"])], textposition="outside", cliponaxis=False))
    fig.update_layout(title="BNPL use by self-rated credit score, non-response kept as a group (unweighted)"); fig.update_xaxes(range=[0, rate["mean"].max() * 125])
    ui.chart(fig, 340)
with t4:
    st.markdown("Choose options and see what the classifier actually receives. Everything is fitted on the training split only.")
    a, b, c = st.columns(3)
    miss_opt = a.selectbox("Missing credit score", ["'Not reported' category", "Most frequent value"])
    scale = b.toggle("Standardise numeric and ordinal inputs", True)
    derive = c.toggle("Add engineered features", True)
    X = train[cfg.FEATURES]
    fe = pl.FeatureEngineer(derive).fit(X)
    prep = pl.preprocessor(cfg.FEATURES, derive, scale, "category" if miss_opt.startswith("'") else "mode").fit(fe.transform(X))
    Z = pd.DataFrame(prep.transform(fe.transform(X)), columns=prep.get_feature_names_out())
    st.write(f"{X.shape[1]} inputs become **{Z.shape[1]} columns**; missing values left: **{int(Z.isna().sum().sum())}**; "
             f"household size capped at **{fe.hh_cap_:.0f}** (training 99th percentile).")
    col = st.selectbox("Compare one variable before and after", cfg.NUMERIC + list(cfg.ORDINALS), format_func=cfg.LABELS.get)
    before = X[col] if col in cfg.NUMERIC else X[col].map({v: i for i, v in enumerate(cfg.ORDINALS[col])})
    a, b = st.columns(2)
    for holder, s, ttl, colr, k in [(a, before, "Before", cfg.C["non"], "bf"), (b, Z[col], "After", cfg.C["user"], "af")]:
        fig = go.Figure(go.Histogram(x=s, marker_color=colr, nbinsx=30))
        fig.update_layout(title=f"{ttl}: mean {s.mean():.2f}, SD {s.std():.2f}")
        with holder:
            ui.chart(fig, 300, k)
    st.dataframe(Z.head(8).round(3), hide_index=True, width="stretch")
with t5:
    y = train[cfg.TARGET]
    strat = st.selectbox("Balancing strategy applied to the training split", pl.STRATEGIES)
    n0, n1 = int((y == 0).sum()), int(y.sum())
    if strat in ("Random oversampling", "SMOTENC"):
        after = (n0, n0, f"Training set grows from {n0 + n1:,} to {2 * n0:,} rows. "
                 + ("Minority rows are repeated." if strat.startswith("Random") else "New minority rows are interpolated between neighbours."))
    elif strat == "Class weights":
        after = (n0, n1, f"Row counts stay the same; each BNPL user counts {n0 / n1:.2f} times as much as a non-user in the loss.")
    else:
        after = (n0, n1, "No change; probabilities keep their natural scale.")
    fig = go.Figure([go.Bar(name="Before", x=["Non-user", "User"], y=[n0, n1], marker_color=cfg.C["non"], text=[n0, n1]),
                     go.Bar(name="After", x=["Non-user", "User"], y=after[:2], marker_color=cfg.C["user"], text=after[:2])])
    fig.update_layout(title="Training rows by class", barmode="group"); ui.chart(fig, 320)
    st.info(after[2] + " Validation and test rows are never resampled.")
    st.dataframe(ui.fmt(ui.table("imbalance_experiments"), dec=3), hide_index=True, width="stretch")
