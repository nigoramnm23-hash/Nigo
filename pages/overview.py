import numpy as np
import plotly.graph_objects as go
import streamlit as st

from bnpl import config as cfg, ui

ui.page("Who uses Buy Now, Pay Later?",
        "An analysis of 12,903 US adults that describes who reports using Buy Now, Pay Later (BNPL) and how well "
        "financial and demographic characteristics identify them.", "Consumer Behaviour Analytics & Machine Learning Platform")

df, m = ui.clean(), ui.meta()
name, _, t = ui.final()
test = ui.table("test_metrics").set_index("Model").loc[name]
w = np.average(df[cfg.TARGET], weights=df["weight"])
ui.kpis([
    ("BNPL use, US adults", f"{w:.1%}", "survey-weighted, past 12 months", True),
    ("Respondents", f"{len(df):,}", f"{int(df[cfg.TARGET].sum()):,} BNPL users (unweighted {df[cfg.TARGET].mean():.1%})"),
    ("Final model", name, f"{len(cfg.FEATURES)} inputs, threshold {t:.2f}"),
    ("Average precision (test)", f"{test['AP (PR-AUC)']:.3f}", f"no-skill baseline {df[cfg.TARGET].mean():.3f}"),
    ("ROC-AUC (test)", f"{test['ROC-AUC']:.3f}", "0.5 = chance, 1.0 = perfect"),
])

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Research question", anchor=False)
    st.markdown("To what extent can financial and demographic characteristics identify US adults who report having used "
                "BNPL in the past year, and which characteristics are most strongly associated with that use?")
    st.subheader("Business problem", anchor=False)
    st.markdown("Providers, regulators and consumer advisers need to understand who relies on BNPL. Only about one adult in six "
                "uses it, so the useful question is how well a model can find that minority and what it costs in false alarms. "
                "The outputs support research and consumer-protection planning. They are not for credit decisions, "
                "pricing or targeting vulnerable people.")
    st.subheader("Data source", anchor=False)
    st.markdown("Board of Governors of the Federal Reserve System, "
                "[Survey of Household Economics and Decisionmaking (SHED) 2025](https://www.federalreserve.gov/consumerscommunities/shed_data.htm), "
                "fielded in October 2025 through a probability-based online panel. Target: question BNPL1, "
                "use of a BNPL service to buy something in the past year.")
with right:
    n = df[cfg.TARGET].value_counts()
    fig = go.Figure(go.Bar(x=[n[1], n[0]], y=["Used BNPL", "Did not use"], orientation="h",
                           marker_color=[cfg.C["user"], cfg.C["non"]],
                           text=[f"{n[1]:,} ({n[1] / len(df):.1%})", f"{n[0]:,} ({n[0] / len(df):.1%})"],
                           textposition=["outside", "inside"], hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_layout(title="Respondents by BNPL use (unweighted)", showlegend=False, bargap=.4)
    fig.update_yaxes(autorange="reversed")
    ui.chart(fig, 250)

st.subheader("Main findings", anchor=False)
perm = ui.table("permutation_importance")
abl = ui.table("ablation")
sel = ui.table("model_selection")
fair = ui.table("fairness_audit")
r = fair[fair.Dimension == "Race/ethnicity"]
best = sel.loc[sel["CV AP"].idxmax()]
st.markdown(f"""
- **Use is concentrated among people under financial and credit pressure.** The three inputs the final model relies on most are
  {", ".join(perm.Label[:3].str.lower()).replace("$", chr(92) + "$")}.
- **Demographics alone are not enough.** Cross-validated average precision rises from {abl['CV AP'].iloc[0]:.3f} with demographics only
  to {abl['CV AP'].iloc[4]:.3f} with all 26 inputs.
- **More complex models add little.** The best cross-validated score was {best['Model']} ({best['CV AP']:.3f}); the selected
  {name} scored {sel.loc[sel.Selected, 'CV AP'].iloc[0]:.3f}, within one standard error, and is the simplest model that close.
- **The model finds {test['Recall (TPR)']:.0%} of BNPL users in unseen data**, and {test['Precision']:.0%} of the people it flags are users.
- **Accuracy would mislead.** Always answering "non-user" is {1 - df[cfg.TARGET].mean():.1%} accurate and finds nobody.
- **Performance differs between groups.** Test recall by race/ethnicity runs from {r['Recall (TPR)'].min():.0%} to {r['Recall (TPR)'].max():.0%},
  although race is not an input.
""")
ui.limits("<b>Read this first.</b> SHED is a cross-sectional survey of self-reports. The model describes who reported <i>past</i> use. It does not "
          "forecast future adoption, and its associations are not causes. It is not a credit-risk, eligibility or approval tool.")
st.write("")
st.markdown("**Where next**")
for c, (p, l, d) in zip(st.columns(3), [("pages/analytics.py", "Consumer analytics", "Filter respondents and compare segments."),
                                        ("pages/performance.py", "Model performance", "See how the models compare on unseen data."),
                                        ("pages/predictor.py", "BNPL predictor", "Score a profile and see why.")]):
    c.page_link(p, label=l, icon=":material/arrow_forward:")
    c.caption(d)
