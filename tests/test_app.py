"""Headless tests of every Streamlit page with streamlit.testing (no browser needed)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from bnpl import config as cfg, pipeline as pl
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["overview", "analytics", "preparation", "laboratory", "performance", "predictor", "simulator"]
META = json.loads((cfg.MODELS / "metadata.json").read_text())
FINAL = META["final_model"]


def open_page(name):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    if name != "overview":
        at.switch_page(f"pages/{name}.py").run()
    assert not at.exception, at.exception
    return at


def text(at):
    return " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("name", PAGES)
def test_page_loads_without_error(name):
    at = open_page(name)
    assert len(at.title) == 1


def test_overview_shows_the_saved_test_metrics():
    at = open_page("overview")
    r = pd.read_csv(cfg.TABLES / "test_metrics.csv").set_index("Model").loc[FINAL]
    assert f"{r['AP (PR-AUC)']:.3f}" in text(at) and f"{r['ROC-AUC']:.3f}" in text(at)


def test_filters_change_the_sample_and_empty_result_is_handled(df):
    at = open_page("analytics")
    at.multiselect[0].set_value(["18-29"]).run()
    assert f"{int(df.age.between(18, 29).sum()):,}" in text(at)
    at.multiselect[3].set_value(["Retired"]).run()            # nobody under 30 is coded as retired
    assert not at.exception and at.warning


def test_weighted_toggle(df):
    at = open_page("analytics")
    at.segmented_control[0].set_value("Unweighted").run()
    assert not at.exception


def test_preparation_controls():
    at = open_page("preparation")
    at.toggle[0].set_value(False).run()
    at.toggle[1].set_value(False).run()
    at.selectbox[0].set_value("Most frequent value").run()
    assert not at.exception
    for s in pl.STRATEGIES:
        at.selectbox[2].set_value(s).run()
        assert not at.exception, s


@pytest.mark.parametrize("name", pl.MODEL_NAMES)
def test_laboratory_trains_every_algorithm(name):
    at = open_page("laboratory")
    at.segmented_control[0].set_value(name).run()
    at.slider[-1].set_value(2000)
    at.button[0].click().run()
    assert not at.exception
    assert "Validation average precision" in text(at)


@pytest.mark.parametrize("strategy", pl.STRATEGIES)
def test_laboratory_strategies(strategy):
    at = open_page("laboratory")
    at.selectbox[-1].set_value(strategy)
    at.slider[-1].set_value(1000)
    at.button[0].click().run()
    assert not at.exception


def test_performance_controls():
    at = open_page("performance")
    at.segmented_control[0].set_value("Survey-weighted").run()
    for m in ["Accuracy", "Brier"]:
        at.selectbox[0].set_value(m).run()
    for v in at.selectbox[1].options:
        at.selectbox[1].set_value(v).run()
        assert not at.exception, v
    at.slider[0].set_value(0.9).run()
    at.slider[0].set_value(0.02).run()
    for v in ["Race/ethnicity", "Age group", "Income group"]:
        at.segmented_control[1].set_value(v).run()
        assert not at.exception, v


def test_predictor_gives_the_model_probability(df):
    import joblib
    from bnpl import explain, ui
    at = open_page("predictor")
    at.button[0].click().run()
    assert not at.exception
    pipe = joblib.load(cfg.MODELS / f"{FINAL.lower().replace(' ', '_')}.joblib")
    assert f"{explain.predict(pipe, ui.typical(df)):.1%}" in text(at)


def test_predictor_rejects_impossible_household():
    at = open_page("predictor")
    n = {w.label: w for w in at.number_input}
    n["Household size"].set_value(1)
    n["Children under 18"].set_value(4)
    at.button[0].click().run()
    assert at.error and not at.exception
    assert "Estimated BNPL propensity" not in text(at)


def test_simulator_presets_and_sweep():
    at = open_page("simulator")
    assert "Difference, A minus B" in text(at)
    for p in at.selectbox[0].options:
        at.selectbox[0].set_value(p).run()
        assert not at.exception and not at.error, p
    sweep = [s for s in at.selectbox if s.label.startswith("Sweep")][0]
    for f in cfg.FEATURES:
        [s for s in at.selectbox if s.label.startswith("Sweep")][0].set_value(f).run()
        assert not at.exception, f
