"""BNPL Intelligence - entry point and navigation. Run with: streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="BNPL Intelligence", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="auto")

pages = {
    "Explore": [
        st.Page("pages/overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("pages/analytics.py", title="Consumer analytics", icon=":material/bar_chart:"),
        st.Page("pages/preparation.py", title="Data preparation", icon=":material/tune:"),
    ],
    "Model": [
        st.Page("pages/laboratory.py", title="ML laboratory", icon=":material/science:"),
        st.Page("pages/performance.py", title="Model performance", icon=":material/monitoring:"),
    ],
    "Use": [
        st.Page("pages/predictor.py", title="BNPL predictor", icon=":material/person_search:"),
        st.Page("pages/simulator.py", title="Consumer simulator", icon=":material/compare_arrows:"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("**BNPL Intelligence**  \nConsumer Behaviour Analytics & Machine Learning Platform")
    st.caption("Data: Federal Reserve Board, SHED 2025. Independent student project, not endorsed by the Board. "
               "Built with AI assistance.")
nav.run()
