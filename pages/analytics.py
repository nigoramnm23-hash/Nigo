import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bnpl import config as cfg, eda, explain, ui

ui.page("Consumer analytics", "Filter the sample and compare how BNPL use varies across consumer segments. Weighted figures estimate US adults; "
        "unweighted figures describe only the respondents.")
df = ui.with_groups(ui.clean())
AGE = ["18-29", "30-44", "45-59", "60+"]
INC = ["Under $25k", "$25k-$50k", "$50k-$100k", "$100k+"]
df["credit_score"] = df["credit_score"].fillna("Not reported")

with st.container(border=True):
    a, b, c, d = st.columns(4)
    f_age = a.multiselect("Age group", AGE, placeholder="All ages")
    f_inc = b.multiselect("Income group", INC, placeholder="All incomes")
    f_wb = c.multiselect("Financial well-being", cfg.WELLBEING, placeholder="All levels")
    f_emp = d.multiselect("Employment", explain.CATEGORIES["employment"], placeholder="All statuses")
    a, b, c, d = st.columns(4)
    f_reg = a.multiselect("Region", explain.CATEGORIES["region"], placeholder="All regions")
    f_card = b.multiselect("Credit-card behaviour", explain.CATEGORIES["card_behaviour"], placeholder="All")
    f_edu = c.multiselect("Education", cfg.EDUCATION, placeholder="All")
    basis = d.segmented_control("Show rates as", ["Weighted", "Unweighted"], default="Weighted", required=True)

sub = df
for col, vals in [("Age group", f_age), ("Income group", f_inc), ("financial_wellbeing", f_wb), ("employment", f_emp),
                  ("region", f_reg), ("card_behaviour", f_card), ("education", f_edu)]:
    if vals:
        sub = sub[sub[col].isin(vals)]
if len(sub) == 0:
    st.warning("No respondents match these filters. Remove one or more filters.")
    st.stop()

k = int(sub[cfg.TARGET].sum())
lo, hi = eda.wilson(k, len(sub))
w = np.average(sub[cfg.TARGET], weights=sub["weight"])
wall = np.average(df[cfg.TARGET], weights=df["weight"])
ui.kpis([
    ("Weighted BNPL rate", f"{w:.1%}", f"{(w - wall) * 100:+.1f} points vs all adults ({wall:.1%})", True),
    ("Unweighted BNPL rate", f"{k / len(sub):.1%}", f"95% interval {lo:.1%} to {hi:.1%}"),
    ("Respondents selected", f"{len(sub):,}", f"{len(sub) / len(df):.0%} of the sample"),
    ("BNPL users selected", f"{k:,}", "unweighted count"),
    ("Population share", f"{sub['weight'].sum() / df['weight'].sum():.0%}", "weighted share of US adults"),
])
if len(sub) < 100:
    st.warning(f"Only {len(sub)} respondents match. Rates from small groups are unreliable; read the interval, not the point estimate.")

KEY = "Weighted rate" if basis == "Weighted" else "Unweighted rate"


def rate_chart(by, order, title, key, height=340):
    t = eda.rate_table(sub, by, order)
    v = t[KEY] * 100
    fig = go.Figure(go.Bar(y=t[by].astype(str), x=v, orientation="h", marker_color=cfg.C["user"],
                           text=[f"{x:.1f}%" if n >= 30 else f"{x:.1f}% (n={n})" for x, n in zip(v, t["n"])], textposition="outside",
                           cliponaxis=False, customdata=np.stack([t["n"], t["Unweighted rate"] * 100, t["Weighted rate"] * 100, t["CI low"] * 100, t["CI high"] * 100], -1),
                           hovertemplate="%{y}<br>n = %{customdata[0]:,}<br>weighted %{customdata[2]:.1f}%<br>unweighted %{customdata[1]:.1f}% (95% interval %{customdata[3]:.1f} to %{customdata[4]:.1f})<extra></extra>"))
    fig.update_layout(title=f"{title} ({basis.lower()})", bargap=.35)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="BNPL use (%)", range=[0, max(10, v.max() * 1.25)])
    ui.chart(fig, height, key)


tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Financial well-being", "Segments", "Data table"])
with tab1:
    a, b = st.columns(2)
    with a:
        rate_chart("Age group", AGE, "BNPL use by age", "age")
        rate_chart("partner_status", explain.CATEGORIES["partner_status"], "By partnership status", "partner")
    with b:
        rate_chart("Income group", INC, "BNPL use by annual income", "inc")
        rate_chart("education", cfg.EDUCATION, "By education", "edu")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for v, col, lab in [(0, cfg.C["non"], "Did not use"), (1, cfg.C["user"], "Used BNPL")]:
            fig.add_histogram(x=sub.loc[sub[cfg.TARGET] == v, "age"], histnorm="percent", xbins=dict(start=18, end=101, size=5),
                              name=lab, marker_color=col, opacity=.72)
        fig.update_layout(barmode="overlay", title="Age distribution (share of each group, unweighted)")
        fig.update_xaxes(title="Age"); fig.update_yaxes(title="% of group")
        ui.chart(fig, 360, "agehist")
    with b:
        t = sub.groupby(["income_band", cfg.TARGET]).size().unstack(fill_value=0).reindex(cfg.INCOME_BANDS, fill_value=0)
        fig = go.Figure()
        for v, col, lab in [(0, cfg.C["non"], "Did not use"), (1, cfg.C["user"], "Used BNPL")]:
            if v in t:
                fig.add_bar(x=t.index, y=t[v] / max(t[v].sum(), 1) * 100, name=lab, marker_color=col)
        fig.update_layout(title="Income distribution (share of each group, unweighted)", barmode="group")
        fig.update_xaxes(tickangle=-45); fig.update_yaxes(title="% of group")
        ui.chart(fig, 360, "inchist")
with tab2:
    a, b = st.columns(2)
    with a:
        rate_chart("financial_wellbeing", cfg.WELLBEING, "By financial well-being", "wb")
        rate_chart("card_behaviour", explain.CATEGORIES["card_behaviour"], "By credit-card behaviour", "card")
        rate_chart("emergency_fund", cfg.YESNO, "By three-month emergency fund", "ef", 260)
    with b:
        rate_chart("bills_status", explain.CATEGORIES["bills_status"], "By bill payment last month", "bills")
        rate_chart("credit_score", explain.CATEGORIES["credit_score"], "By self-rated credit score", "cs")
        rate_chart("can_cover_400", cfg.YESNO, "Can cover a $400 expense", "c400", 260)
with tab3:
    g = sub.pivot_table(index="financial_wellbeing", columns="Income group", values=cfg.TARGET, aggfunc=["mean", "size"])
    r = g["mean"].reindex(index=cfg.WELLBEING, columns=INC) * 100
    n = g["size"].reindex(index=cfg.WELLBEING, columns=INC)
    txt = [["" if pd.isna(r.iloc[i, j]) else f"{r.iloc[i, j]:.0f}%<br>n={int(n.iloc[i, j])}" for j in range(4)] for i in range(4)]
    fig = go.Figure(go.Heatmap(z=r, x=INC, y=cfg.WELLBEING, colorscale=ui.EMERALD, text=txt, texttemplate="%{text}",
                               colorbar=dict(title="%"), hovertemplate="%{y}, %{x}: %{z:.1f}%<extra></extra>"))
    fig.update_layout(title="BNPL use by income and well-being (unweighted)")
    ui.chart(fig, 380, "seg")
    st.markdown("**Build a segment comparison**")
    a, b = st.columns(2)
    opts = [f for f in cfg.FEATURES if f not in cfg.NUMERIC and f != "income_band"] + ["Age group", "Income group"]
    lab = lambda f: cfg.LABELS.get(f, f)
    d1 = a.selectbox("Segment by", opts, format_func=lab, index=opts.index("credit_confidence"))
    d2 = b.selectbox("Split by", [o for o in opts if o != d1], format_func=lab)
    seg = sub.groupby([d1, d2]).apply(lambda s: pd.Series({"Respondents": len(s), "Unweighted rate": s[cfg.TARGET].mean(),
                                     "Weighted rate": np.average(s[cfg.TARGET], weights=s["weight"])}), include_groups=False).reset_index()
    seg["Respondents"] = seg["Respondents"].astype(int)
    seg = seg.sort_values(KEY, ascending=False)
    st.dataframe(ui.fmt(seg.rename(columns={d1: lab(d1), d2: lab(d2)}), pct=("Unweighted rate", "Weighted rate")), hide_index=True, width="stretch")
with tab4:
    cols = ["Age group", "Income group", "financial_wellbeing", "education", "employment", "card_behaviour", cfg.TARGET]
    st.caption(f"First 500 of {len(sub):,} matching respondents (de-identified public-use data).")
    st.dataframe(sub[cols].rename(columns=cfg.LABELS).head(500), hide_index=True, width="stretch")
