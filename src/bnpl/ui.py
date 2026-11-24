"""Shared Streamlit helpers: styling, cached loaders, KPI cards and Plotly defaults."""
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from . import config as cfg
from . import data, explain, experiments as ex, pipeline as pl

C = cfg.C
EMERALD = [[0, "#F2F7F4"], [1, C["user"]]]
pio.templates["bnpl"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color=C["ink"]), paper_bgcolor="white",
    plot_bgcolor="white", margin=dict(l=12, r=20, t=76, b=12), title=dict(font=dict(size=15), x=0.01, xanchor="left", y=0.97, yanchor="top"),
    xaxis=dict(gridcolor=C["grid"], zeroline=False, automargin=True, title=dict(font=dict(color=C["muted"]))),
    yaxis=dict(gridcolor=C["grid"], zeroline=False, automargin=True, title=dict(font=dict(color=C["muted"]))),
    legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), hoverlabel=dict(bgcolor="white", font=dict(color=C["ink"])),
    colorway=[C["user"], "#2A78D6", "#EB6834", "#6250D6"]))
pio.templates.default = "bnpl"

CSS = """
<style>
.block-container {padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1200px;}
h1 {font-weight: 700; letter-spacing: -0.02em;} h2, h3 {letter-spacing: -0.01em;}
.eyebrow {color:#0B6E4F; font-size:.8rem; font-weight:600; margin-bottom:.1rem;}
.lede {color:#4A5952; font-size:1.02rem; max-width: 78ch; margin:.1rem 0 1.1rem 0;}
.kpi {background:#fff; border:1px solid #E1E4DC; border-radius:10px; padding:14px 16px; min-height:112px;}
.kpi .l {color:#5B6B64; font-size:.82rem;} .kpi .v {font-size:1.65rem; font-weight:700; color:#14241E; line-height:1.25;}
.kpi .n {color:#5B6B64; font-size:.78rem;}
.kpi.hero {background:#0B6E4F; border-color:#0B6E4F;} .kpi.hero * {color:#fff !important;}
[data-testid="stSidebar"] {border-right:1px solid #E1E4DC;}
[data-testid="stPlotlyChart"] {background:#fff; border:1px solid #E1E4DC; border-radius:10px; overflow:hidden;}
.limits {background:#fff; border:1px solid #E1E4DC; border-left:4px solid #0B6E4F; border-radius:8px; padding:12px 16px; color:#33433B; font-size:.92rem;}
</style>
"""


def page(title, lede, eyebrow="BNPL Intelligence"):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f"<div class='lede'>{lede}</div>", unsafe_allow_html=True)


def kpis(items):
    for col, (label, value, note, *hero) in zip(st.columns(len(items)), items):
        col.markdown(f"<div class='kpi{' hero' if hero else ''}'><div class='l'>{label}</div>"
                     f"<div class='v'>{value}</div><div class='n'>{note}</div></div>", unsafe_allow_html=True)
    st.write("")


def limits(text):
    st.markdown(f"<div class='limits'>{text}</div>", unsafe_allow_html=True)


def chart(fig, height=380, key=None):
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", key=key, theme=None, config={"displayModeBar": False})


@st.cache_data(show_spinner=False)
def clean():
    return data.load_clean()


@st.cache_data(show_spinner=False)
def table(name):
    return pd.read_csv(cfg.TABLES / f"{name}.csv")


@st.cache_data(show_spinner=False)
def meta():
    return json.loads((cfg.MODELS / "metadata.json").read_text())


@st.cache_resource(show_spinner=False)
def model(name):
    return joblib.load(cfg.MODELS / f"{name.lower().replace(' ', '_')}.joblib")


@st.cache_data(show_spinner=False)
def splits():
    return pl.split(clean())


@st.cache_resource(show_spinner=False)
def explainer_parts(name):
    """Background mean for linear models; TreeExplainer for tree ensembles."""
    pipe = model(name)
    train, valid, _ = splits()
    bg = explain.background(pipe, pd.concat([train, valid])[cfg.FEATURES])
    if hasattr(pipe.named_steps["clf"], "coef_"):
        return bg, None
    import shap
    return bg, shap.TreeExplainer(pipe.named_steps["clf"])


def final():
    m = meta()
    name = m["final_model"]
    return name, model(name), m["thresholds"][name]


def with_groups(df):
    return ex.add_groups(df)


def fmt(df, pct=(), dec=3):
    d = df.copy()
    for c in d.columns:
        if c in pct:
            d[c] = (d[c] * 100).map(lambda v: "-" if pd.isna(v) else f"{v:.1f}%")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(dec)
    return d


def typical(df):
    """Most common / median value of every model input (a data-driven starting profile)."""
    p = {}
    for f in cfg.FEATURES:
        if f in cfg.NUMERIC:
            p[f] = int(df[f].median())
        else:
            p[f] = df[f].fillna("Not reported").mode()[0]
    p["household_size"] = max(p["household_size"], 1 + p["children_u18"])
    return p


def field(f, value, key):
    """One input widget whose options match the trained model exactly."""
    label = cfg.LABELS[f]
    if f in explain.RANGES:
        lo, hi = explain.RANGES[f]
        return int(st.number_input(label, lo, hi, int(value), step=1, key=key))
    cats = explain.CATEGORIES[f]
    if cats == cfg.YESNO:
        return st.segmented_control(label, cats, default=value, key=key, required=True)
    return st.selectbox(label, cats, index=cats.index(value), key=key)
