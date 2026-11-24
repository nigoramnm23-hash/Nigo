"""Data integrity, leakage prevention and preprocessing."""
import numpy as np
import pandas as pd
import pytest
from bnpl import config as cfg, data, eda, pipeline as pl


def test_raw_file_is_the_official_download(raw):
    assert data.raw_sha256() == cfg.RAW_SHA256
    assert raw.shape == (12934, 815)
    assert raw["year"].unique().tolist() == [2025]
    assert raw["BNPL1"].value_counts().to_dict() == {"No": 10930, "Yes": 2004}


def test_cleaning_is_reproducible_and_matches_saved_table(raw, df):
    again, log = data.clean(raw[data.RAW_COLS])
    assert len(again) == len(df) == 12903
    assert log.set_index("Check")["Rows affected"].iloc[1] == 31
    for c in cfg.FEATURES + [cfg.TARGET]:
        a, b = again[c].fillna("NA").astype(str).tolist(), df[c].fillna("NA").astype(str).tolist()
        if c in cfg.NUMERIC:
            a, b = again[c].astype(float).tolist(), df[c].astype(float).tolist()
        assert a == b, c


def test_labels_and_ranges(df):
    for c, cats in cfg.ORDINALS.items():
        assert set(df[c].dropna()) <= set(cats), c
    assert df["shedid"].is_unique
    assert df["age"].between(18, 98).all()
    assert (df["household_size"] >= 1 + df["children_u18"]).all()
    assert df["afs_count"].between(0, 5).all() and df["medical_skips"].between(0, 5).all()


def test_weighted_rates_match_published_fed_figures(df):
    v = eda.fed_validation(df)
    assert len(v) == 15
    assert v["Difference"].abs().max() <= 1.0


def test_no_leaky_or_protected_inputs():
    banned = set(data.BNPL_FOLLOWUPS) | {"BNPL1", "BNPL1_iflag", "weight", "shedid", cfg.TARGET, *cfg.PROTECTED}
    assert not banned & set(cfg.FEATURES)
    assert not any(c in data.RAW_COLS for c in data.BNPL_FOLLOWUPS)
    assert len(cfg.FEATURES) == len(set(cfg.FEATURES)) == 26


def test_followups_would_leak(raw):
    """Evidence for excluding them: they are answered by BNPL users only."""
    users = raw["BNPL1"].eq("Yes")
    assert raw.loc[users, "BNPL3"].notna().all()
    assert raw.loc[~users, data.BNPL_FOLLOWUPS].isna().all().all()


def test_split_is_disjoint_stratified_and_reproducible(df, splits):
    train, valid, test = splits
    ids = [set(s["shedid"]) for s in splits]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert sum(map(len, ids)) == len(df)
    assert all(abs(s[cfg.TARGET].mean() - df[cfg.TARGET].mean()) < 0.002 for s in splits)
    assert [s.index.tolist() for s in pl.split(df)] == [s.index.tolist() for s in splits]
    assert abs(len(test) / len(df) - 0.20) < 0.001


def test_saved_test_predictions_are_exactly_the_test_rows(splits):
    saved = pd.read_csv(cfg.TABLES / "test_predictions.csv", dtype={"shedid": str})
    assert set(saved["shedid"]) == set(splits[2]["shedid"])


def test_preprocessor_learns_from_training_only(splits):
    train, _, test = splits
    pipe = pl.build("Logistic Regression").fit(*pl.xy(train))
    num = pipe.named_steps["prep"].named_transformers_["num"]
    fe = pipe.named_steps["engineer"]
    assert fe.hh_cap_ == np.nanpercentile(train["household_size"], 99)
    cols = pipe.named_steps["prep"].transformers_[0][2]
    expected = fe.transform(train[cfg.FEATURES])[cols].astype(float).mean().to_numpy()
    assert np.allclose(num.named_steps["scale"].mean_, expected)
    changed = test[cfg.FEATURES].copy()
    changed["age"] = 18
    before = pipe.predict_proba(train[cfg.FEATURES].head(50))[:, 1]
    pipe.predict_proba(changed)                         # scoring other rows must not change the pipeline
    assert np.array_equal(before, pipe.predict_proba(train[cfg.FEATURES].head(50))[:, 1])


def test_no_missing_values_reach_the_model(splits):
    train, valid, test = splits
    pipe = pl.build("Logistic Regression").fit(*pl.xy(train))
    for s in (train, valid, test):
        assert not np.isnan(np.asarray(pipe[:-1].transform(s[cfg.FEATURES]), dtype=float)).any()


@pytest.mark.parametrize("strategy", pl.STRATEGIES)
def test_resampling_happens_in_fit_only(splits, strategy):
    train, valid, _ = splits
    small = train.sample(1500, random_state=1)
    pipe = pl.build("Logistic Regression", {"C": 0.1}, strategy).fit(*pl.xy(small))
    p = pipe.predict_proba(valid[cfg.FEATURES])
    assert p.shape == (len(valid), 2) and np.allclose(p.sum(1), 1)


def test_unseen_category_and_missing_value_do_not_crash(splits):
    pipe = pl.build("Logistic Regression").fit(*pl.xy(splits[0]))
    row = splits[2][cfg.FEATURES].head(1).copy()
    row["region"] = "Atlantis"
    row["education"] = np.nan
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1
