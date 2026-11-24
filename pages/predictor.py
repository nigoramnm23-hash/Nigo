import plotly.graph_objects as go
import streamlit as st

from bnpl import config as cfg, explain, ui

ui.page("BNPL predictor", "Enter a profile to see the estimated propensity to report BNPL use, how it is classified, and which answers moved the estimate.")
name, pipe, t = ui.final()
df = ui.clean()
start = ui.typical(df)
bg, tex = ui.explainer_parts(name)

with st.form("predict"):
    profile = {}
    tabs = st.tabs(list(cfg.GROUPS))
    for tab, (g, feats) in zip(tabs, cfg.GROUPS.items()):
        with tab:
            cols = st.columns(3)
            for i, f in enumerate(feats):
                with cols[i % 3]:
                    profile[f] = ui.field(f, start[f], f"p_{f}")
    st.caption("Starting values are the most common answers in the sample. Every field corresponds to one input of the trained model.")
    go_btn = st.form_submit_button("Predict", type="primary", icon=":material/insights:")

if go_btn:
    problems = explain.validate(profile)
    if problems:
        st.session_state.pop("pred", None)
        for p in problems:
            st.error(p)
        st.stop()
    st.session_state["pred"] = dict(profile)
if "pred" not in st.session_state:
    st.info("Fill in the five tabs and select Predict.")
    st.stop()

profile = st.session_state["pred"]
p = explain.predict(pipe, profile)
tp = ui.table("test_predictions")
near = tp[(tp[name] - p).abs() <= .03]
label = "Likely BNPL user" if p >= t else "Unlikely BNPL user"
ui.kpis([("Estimated BNPL propensity", f"{p:.1%}", f"{name}, sample average {df[cfg.TARGET].mean():.1%}", True),
         ("Classification", label, f"threshold {t:.0%}, chosen on validation data"),
         ("Observed rate at similar scores", f"{near['y'].mean():.1%}" if len(near) >= 30 else "Too few cases", f"{len(near):,} test respondents within 3 points")])
tab, base = explain.contributions(pipe, profile, bg, tex)
d = tab.head(12).iloc[::-1]
fig = go.Figure(go.Bar(x=d.phi, y=[f"{l}: {v}" if v != "derived" else l for l, v in zip(d.Label, d.Value)], orientation="h",
                       marker_color=[cfg.C["user"] if v > 0 else cfg.C["accent"] for v in d.phi], hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>"))
fig.update_layout(title="What moved this estimate, compared with the average respondent")
fig.update_xaxes(title="Contribution in log-odds (right raises, left lowers)")
ui.chart(fig, 460)
up, down = tab[tab.phi > 0].head(3), tab[tab.phi < 0].head(3)
words = (f"**In words.** Raising the estimate most: {', '.join(up.Label.str.lower()) or 'nothing'}. "
         f"Lowering it most: {', '.join(down.Label.str.lower()) or 'nothing'}.")
st.markdown(words.replace("$", "\\$"))      # a bare dollar sign would be read as LaTeX
ui.limits("<b>Limits of this estimate.</b> It is the model's estimate that someone with this profile would <i>report</i> past-year BNPL use in a survey like SHED 2025. "
          "Most people above the threshold are still non-users, because BNPL use is uncommon. It says nothing about future adoption, ability to repay or creditworthiness, "
          "must not be used to approve, price or target credit, and the bars show association, not cause. Nothing you enter is stored.")
