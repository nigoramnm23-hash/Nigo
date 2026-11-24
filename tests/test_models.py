"""Saved models, saved metrics, selection rules and single-profile prediction."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from bnpl import config as cfg, experiments as ex, explain, pipeline as pl, ui

META = json.loads((cfg.MODELS / "metadata.json").read_text())
FINAL = META["final_model"]


def load(name):
    return joblib.load(cfg.MODELS / f"{name.lower().replace(' ', '_')}.joblib")


@pytest.mark.parametrize("name", pl.MODEL_NAMES + ["Dummy"])
def test_saved_models_reproduce_saved_test_predictions(name, splits):
    saved = pd.read_csv(cfg.TABLES / "test_predictions.csv", dtype={"shedid": str}).set_index("shedid")
    test = splits[2]
    p = load(name).predict_proba(test[cfg.FEATURES])[:, 1]
    assert np.allclose(p, saved.loc[test["shedid"], name], atol=1e-9)


@pytest.mark.parametrize("name", pl.MODEL_NAMES + ["Dummy"])
def test_saved_metrics_are_recomputable(name):
    res = pd.read_csv(cfg.TABLES / "test_metrics.csv").set_index("Model")
    pr = pd.read_csv(cfg.TABLES / "test_predictions.csv")
    m = ex.metrics(pr["y"].to_numpy(), pr[name].to_numpy(), META["thresholds"].get(name, 0.5))
    for k in ["Accuracy", "Precision", "Recall (TPR)", "F1", "ROC-AUC", "AP (PR-AUC)", "Brier", "TP", "FP", "FN", "TN"]:
        assert np.isclose(m[k], res.loc[name, k], atol=1e-9), k


def test_final_model_retrains_to_the_same_predictions(splits):
    train, valid, test = splits
    dev = pd.concat([train, valid])
    again = pl.build(FINAL, META["best_params"][FINAL], META["strategy"][FINAL]).fit(*pl.xy(dev))
    assert np.allclose(again.predict_proba(test[cfg.FEATURES])[:, 1], load(FINAL).predict_proba(test[cfg.FEATURES])[:, 1], atol=1e-9)


def test_threshold_comes_from_validation_data_only(splits):
    train, valid, _ = splits
    p = pl.build(FINAL, META["best_params"][FINAL], META["strategy"][FINAL]).fit(*pl.xy(train)).predict_proba(valid[cfg.FEATURES])[:, 1]
    assert ex.best_threshold(valid[cfg.TARGET].to_numpy(), p) == META["thresholds"][FINAL]


def test_selection_follows_the_written_protocol():
    imb = pd.read_csv(cfg.TABLES / "imbalance_experiments.csv")
    sel, strategies, final = ex.choose(imb)
    assert final == FINAL and strategies == META["strategy"]
    best = sel.loc[sel["CV AP"].idxmax()]
    chosen = sel[sel.Selected].iloc[0]
    assert chosen["CV AP"] >= best["CV AP"] - best["CV AP SE"]


def test_models_beat_the_baseline():
    res = pd.read_csv(cfg.TABLES / "test_metrics.csv").set_index("Model")
    for n in pl.MODEL_NAMES:
        assert res.loc[n, "AP (PR-AUC)"] > 2 * res.loc["Dummy", "AP (PR-AUC)"]
        assert res.loc[n, "Brier"] < res.loc["Dummy", "Brier"]


@pytest.fixture(scope="module")
def typical(df):
    return ui.typical(df)


def test_prediction_is_deterministic_and_bounded(typical):
    pipe = load(FINAL)
    p = explain.predict(pipe, typical)
    assert 0 < p < 1 and p == explain.predict(pipe, typical)


def test_every_allowed_category_can_be_scored(typical):
    pipe = load(FINAL)
    for f, cats in explain.CATEGORIES.items():
        for c in cats:
            prof = {**typical, f: c}
            if not explain.validate(prof):
                assert 0 < explain.predict(pipe, prof) < 1


def test_training_categories_equal_app_categories(df):
    for f, cats in explain.CATEGORIES.items():
        seen = set(df[f].fillna("Not reported") if f == "credit_score" else df[f].dropna())
        assert seen == set(cats), f


@pytest.mark.parametrize("change", [
    {"age": 17}, {"age": 150}, {"age": -5}, {"age": 30.5}, {"age": float("nan")}, {"age": "forty"}, {"age": None}, {"age": True},
    {"household_size": 0}, {"household_size": 1, "children_u18": 3}, {"income_band": "a million"}, {"region": ""},
    {"employment": "Retired", "age": 20}, {"afs_count": 6}, {"household_size": 1, "partner_status": "Living with partner"}])
def test_invalid_profiles_are_rejected(typical, change):
    prof = {**typical, **change}
    assert explain.validate(prof)
    with pytest.raises(ValueError):
        explain.predict(load(FINAL), prof)


def test_missing_field_is_rejected(typical):
    prof = dict(typical)
    prof.pop("age")
    assert explain.validate(prof)


def test_extreme_but_valid_profiles(typical):
    pipe = load(FINAL)
    hi = {**typical, "age": 98, "household_size": 12, "children_u18": 9, "afs_count": 5, "medical_skips": 5}
    lo = {**typical, "age": 18, "household_size": 1, "children_u18": 0, "afs_count": 0, "medical_skips": 0, "partner_status": "Never married",
          "education": cfg.EDUCATION[1], "income_band": cfg.INCOME_BANDS[0]}
    for prof in (hi, lo):
        assert 0 < explain.predict(pipe, prof) < 1


def test_contributions_add_up_to_the_prediction(typical, splits):
    pipe = load(FINAL)
    bg = explain.background(pipe, pd.concat(splits[:2])[cfg.FEATURES])
    t, base = explain.contributions(pipe, typical, bg)
    p = 1 / (1 + np.exp(-(base + t.phi.sum())))
    assert np.isclose(p, explain.predict(pipe, typical))
