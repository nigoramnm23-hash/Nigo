import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bnpl import config as cfg, explain, ui

ui.page("Consumer simulator", "Compare two hypothetical consumers and see how the model responds when characteristics change.")
name, pipe, t = ui.final()
df = ui.clean()
bg, tex = ui.explainer_parts(name)
typ = ui.typical(df)
PRESETS = {
    "Typical respondent": {},
    "Financially strained renter": {"age": 32, "household_size": 3, "children_u18": 1, "income_band": "$35,000 to $39,999", "education": cfg.EDUCATION[1],
        "financial_wellbeing": "Just getting by", "change_vs_last_year": "Somewhat worse off", "emergency_fund": "No", "can_cover_400": "No",
        "bills_status": "Paid in full with difficulty", "card_behaviour": "Carries balance mostly", "credit_score": "Fair",
        "credit_confidence": "Not confident", "housing": "Renter", "partner_status": "Never married", "retirement_account": "No",
        "spending_vs_income": "More than your income", "food_security": cfg.FOOD[2]},
    "Comfortable retiree": {"age": 70, "household_size": 2, "children_u18": 0, "employment": "Retired", "financial_wellbeing": "Living comfortably",
        "emergency_fund": "Yes", "can_cover_400": "Yes", "housing": "Owner, no mortgage", "card_behaviour": "Pays in full", "credit_score": "Excellent",
        "credit_confidence": "Very confident", "spending_vs_income": "Less than your income"},
    "Young graduate with student loans": {"age": 25, "household_size": 1, "children_u18": 0, "income_band": "$50,000 to $59,999", "education": cfg.EDUCATION[3],
        "student_loan": "Yes", "housing": "Renter", "partner_status": "Never married", "card_behaviour": "Carries balance sometimes",
        "emergency_fund": "No", "financial_wellbeing": "Doing okay", "credit_score": "Good", "credit_confidence": "Somewhat confident"},
}
st.caption("Presets are illustrative scenarios written for this app, not real people. Each starts from the typical respondent.")
a, b = st.columns(2)
pa = a.selectbox("Start profile A from", list(PRESETS), index=1)
pb = b.selectbox("Start profile B from", list(PRESETS), index=2)
base = {"A": {**typ, **PRESETS[pa]}, "B": {**typ, **PRESETS[pb]}}
default = ["age", "income_band", "financial_wellbeing", "emergency_fund", "card_behaviour", "credit_confidence"]
edit = st.multiselect("Characteristics to modify", cfg.FEATURES, default=default, format_func=cfg.LABELS.get)
prof = {}
a, b = st.columns(2)
for col, k, preset in [(a, "A", pa), (b, "B", pb)]:
    with col.container(border=True):
        st.markdown(f"**Profile {k}**")
        prof[k] = dict(base[k])
        for f in edit:
            prof[k][f] = ui.field(f, base[k][f], f"s_{k}_{preset}_{f}")
bad = False
for k in "AB":
    for e in explain.validate(prof[k]):
        st.error(f"Profile {k}: {e}"); bad = True
if bad:
    st.stop()

p = {k: explain.predict(pipe, prof[k]) for k in "AB"}
ui.kpis([("Profile A propensity", f"{p['A']:.1%}", "above threshold" if p["A"] >= t else "below threshold", True),
         ("Profile B propensity", f"{p['B']:.1%}", "above threshold" if p["B"] >= t else "below threshold", True),
         ("Difference, A minus B", f"{(p['A'] - p['B']) * 100:+.1f} points", f"threshold {t:.0%}"),
         ("Sample average", f"{df[cfg.TARGET].mean():.1%}", "unweighted BNPL rate")])
ca, _ = explain.contributions(pipe, prof["A"], bg, tex)
cb, _ = explain.contributions(pipe, prof["B"], bg, tex)
d = ca.merge(cb, on=["Feature", "Label"], suffixes=(" A", " B"))
d["diff"] = d["phi A"] - d["phi B"]
d = d.reindex(d["diff"].abs().sort_values(ascending=False).index).head(12).iloc[::-1]
left, right = st.columns(2)
with left:
    fig = go.Figure([go.Bar(y=d.Label, x=d["phi A"], name="Profile A", orientation="h", marker_color=cfg.C["user"]),
                     go.Bar(y=d.Label, x=d["phi B"], name="Profile B", orientation="h", marker_color=cfg.C["accent"])])
    fig.update_layout(title="Where the profiles differ most (log-odds vs average)", barmode="group"); ui.chart(fig, 520)
with right:
    f = st.selectbox("Sweep one characteristic, holding the rest fixed", cfg.FEATURES, index=cfg.FEATURES.index("income_band"), format_func=cfg.LABELS.get)
    vals = list(range(explain.RANGES[f][0], explain.RANGES[f][1] + 1)) if f in explain.RANGES else explain.CATEGORIES[f]
    fig = go.Figure()
    for k, col, dash in [("A", cfg.C["user"], "solid"), ("B", cfg.C["accent"], "dash")]:
        ok = [v for v in vals if not explain.validate({**prof[k], f: v})]
        ys = pipe.predict_proba(pd.concat([explain.frame({**prof[k], f: v}) for v in ok], ignore_index=True))[:, 1] * 100 if ok else []
        fig.add_scatter(x=[str(v) if f not in explain.RANGES else v for v in ok], y=ys, name=f"Profile {k}", mode="lines+markers", line=dict(color=col, dash=dash, width=2), marker_size=8)
    fig.add_hline(y=t * 100, line_color=cfg.C["non"], annotation_text="threshold", annotation_position="top left")
    fig.update_layout(title=f"Propensity as {cfg.LABELS[f].lower()} changes"); fig.update_yaxes(title="Estimated propensity (%)", rangemode="tozero"); fig.update_xaxes(tickangle=-45)
    ui.chart(fig, 440)
diff = [f for f in cfg.FEATURES if prof["A"][f] != prof["B"][f]]
st.markdown(f"**Profiles differ on {len(diff)} of {len(cfg.FEATURES)} inputs**")
st.dataframe(pd.DataFrame({"Characteristic": [cfg.LABELS[f] for f in diff], "Profile A": [str(prof["A"][f]) for f in diff], "Profile B": [str(prof["B"][f]) for f in diff]}), hide_index=True, width="stretch")
ui.limits("<b>Scenarios, not causes.</b> The comparison shows how the model's estimate changes when its inputs change. It does not show what would happen if a real "
          "person's income, savings or credit changed: characteristics move together in reality, and the survey observed each person only once. Some lines run against the raw trend: BNPL use falls with income overall, yet the model's income line can rise, because it asks about higher income <i>with the same</i> credit score, savings and bills.")
