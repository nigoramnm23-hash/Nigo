# BNPL Intelligence

**Consumer Behaviour Analytics & Machine Learning Platform.** Predicts whether a US adult reports having used Buy Now, Pay Later (BNPL) in the past year, from financial and demographic characteristics in the Federal Reserve Board's SHED 2025 survey. Binary classification with imbalanced classes (15.5% positive).

**Live application:** https://bnpl-intelligence.streamlit.app

This is a consumer-behaviour study. It is **not** a credit-risk, eligibility, pricing or targeting tool, and its associations are not causal.

## Results (held-out test set, n = 2,581)

| Model | AP (PR-AUC) | ROC-AUC | Brier | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Dummy (base rate) | 0.155 | 0.500 | 0.131 | 0.50 | 0.00 | 0.00 | 0.00 |
| **Logistic Regression (selected)** | **0.426** | **0.789** | **0.109** | 0.23 | 0.38 | 0.56 | 0.45 |
| Decision Tree | 0.350 | 0.748 | 0.116 | 0.19 | 0.32 | 0.59 | 0.41 |
| Random Forest | 0.426 | 0.784 | 0.111 | 0.21 | 0.34 | 0.61 | 0.44 |
| Gradient Boosting | 0.426 | 0.791 | 0.109 | 0.28 | 0.42 | 0.47 | 0.44 |

Selection rule (fixed before results): the simplest model within one standard error of the best cross-validated AP.

## Quick start

Requires **Python 3.12**.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

With [uv](https://docs.astral.sh/uv/) the exact locked environment (with file hashes) is one command:

```bash
uv sync --locked
uv run streamlit run app.py
```

| Task | Command |
|---|---|
| Run the app | `streamlit run app.py` |
| Run the notebook | open `notebooks/BNPL_Adoption_Prediction.ipynb`, Run All (10-15 min) |
| Retrain everything | `python scripts/train.py` |
| Rebuild EDA outputs | `python scripts/eda.py` |
| Run the tests | `python -m pytest` |
| Run in Docker | `docker build -t bnpl . && docker run -p 8501:8501 bnpl` |

## Structure

```
app.py                  navigation (7 pages in pages/)
src/bnpl/               data.py cleaning | pipeline.py preprocessing + models | experiments.py stages
                        eda.py statistics | plots.py figures | explain.py prediction + explanation | ui.py
notebooks/              BNPL_Adoption_Prediction.ipynb (executed)
scripts/                train.py, eda.py
models/                 trained pipelines (.joblib), metadata.json, best_params.json
data/raw, data/docs     unmodified official ZIP and codebook; data/processed: cleaned table
outputs/tables, figures every number and chart used in the notebook and app
tests/                  73 automated tests: data, leakage, models, app pages, invalid inputs
docs/                   testing report, deployment guide, development log
```

## Reproducibility

- `requirements.txt` pins every package; `uv.lock` adds hashes; `pyproject.toml` fixes Python 3.12; a `Dockerfile` is provided.
- One seed (42) everywhere. A SHA-256 check confirms the raw file. Tests retrain the final model and compare it with the saved one.
- The notebook, scripts and app call the same functions in `src/bnpl`.

## Data and licence

Code: MIT (`LICENSE`). Data: Board of Governors of the Federal Reserve System, SHED 2025; see `DATA_NOTICE.md`. Not endorsed by the Board.

## AI assistance

Developed with substantial assistance from an AI coding assistant (Claude, Anthropic); see `docs/DEVELOPMENT_LOG.md`.
