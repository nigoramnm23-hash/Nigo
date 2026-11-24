"""Recreate every EDA table and figure: python scripts/eda.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from bnpl import config as cfg, data, eda, experiments as ex, plots  # noqa: E402

raw, df = data.load_raw(data.RAW_COLS), data.load_clean()
ex.save(eda.descriptive(df).reset_index(names="Variable"), "descriptive")
ex.save(eda.association(df), "association")
ex.save(eda.fed_validation(df), "fed_validation")
ex.save(eda.spearman_matrix(df), "spearman", index=True)
ex.save(eda.vif(df), "vif")
plots.target(df); plots.missing(raw); plots.histograms(df); plots.boxplots(df); plots.demographics(df)
plots.income(df); plots.finance(df); plots.heat_income_wellbeing(df); plots.age_by_wellbeing(df)
plots.spearman(df); plots.association(eda.association(df))
for t in ["descriptive", "association", "fed_validation", "vif"]:
    print("\n", t); print(__import__("pandas").read_csv(cfg.TABLES / f"{t}.csv").round(4).to_string())
