"""Run every modelling stage end to end: python scripts/train.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from bnpl import config as cfg, data, experiments as ex, pipeline as pl  # noqa: E402


def main():
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    assert data.raw_sha256() == cfg.RAW_SHA256, "Raw dataset differs from the official download"
    df = data.load_clean(refresh=True)
    raw = data.load_raw(data.RAW_COLS)
    fd, exc = data.feature_dictionary(raw, df)
    ex.save(fd, "feature_dictionary")
    ex.save(exc, "excluded_variables")
    train, valid, test = pl.split(df)
    step(f"rows {len(train)}/{len(valid)}/{len(test)}")
    print(ex.ablation(train).round(4).to_string()); step("ablation")
    print(ex.preprocessing_choices(train).round(4).to_string()); step("preprocessing choices")
    summary, best = ex.tune(train); print(summary.round(4).to_string()); print(best); step("tuning")
    imb = ex.imbalance(train, valid, best); print(imb.round(4).to_string()); step("imbalance")
    sel, strategies, final = ex.choose(imb); print(sel.round(4).to_string()); step(f"FINAL MODEL: {final} {strategies}")
    th, vpred = ex.thresholds(train, valid, best, strategies); print(th)
    ex.save(ex.threshold_table(vpred["y"].to_numpy(), vpred[final].to_numpy()), "threshold_table")
    ex.save_metadata(final, strategies, th, best, {"train": train, "valid": valid, "test": test})
    fitted = ex.final_fit(train, valid, best, strategies); step("final fit")
    res, wres, ci, preds = ex.test_evaluation(fitted, test, th)
    print(res.round(4).to_string()); print(ci.round(4).to_string()); step("test evaluation")
    ex.calibration(preds)
    dev = pd.concat([train, valid])
    ex.coefficients(fitted["Logistic Regression"])
    print(ex.permutation(fitted[final], test).head(12).round(4).to_string())
    ex.tree_importance(fitted["Random Forest"])
    print(ex.shap_summary(fitted[final], dev.sample(500, random_state=cfg.SEED), test)[0].head(12).round(4).to_string())
    print(ex.fairness(test, preds, final, th[final]).round(3).to_string()); step("interpretation and fairness")
    print(ex.learning(train, final, best[final], strategies[final]).round(4).to_string())
    print(ex.weight_sensitivity(train, valid, final, best[final], strategies[final]).round(4).to_string())
    step("done")


if __name__ == "__main__":
    main()
