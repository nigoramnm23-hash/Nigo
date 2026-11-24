# Testing report

Date: 24 November 2026. Environment: Windows 10, Python 3.12.10, 4 logical processors. Run again with `python -m pytest`.

**Summary:** 73 of 73 automated tests pass, both in the development environment and in a **clean clone installed from `uv.lock`** (`uv sync --locked`). The notebook ran from beginning to end: 33 of 33 code cells, no errors. Two independent full runs (`scripts/train.py`, then the notebook) gave the same test metrics to four decimals.

| Area | What was tested | Result |
|---|---|---|
| Raw data | SHA-256, shape 12,934 x 815, year 2025, target counts | Pass |
| External validity | 15 survey-weighted rates against the Fed's published Table 41 | Pass, largest difference 0.6 points |
| Cleaning | Re-cleaning reproduces the saved table; valid labels, ranges and household logic | Pass |
| Leakage | No follow-up, weight, identifier or protected input; follow-ups shown to be blank for all non-users | Pass |
| Split | Disjoint, stratified, reproducible, 60/20/20 | Pass |
| Preprocessing | Learned from training rows only; no missing values reach the model; unseen categories handled | Pass |
| Imbalance | Resampling runs in fit only, for all four strategies | Pass |
| Tuning | Tuning functions receive training rows only (code review); saved threshold is reproduced from train + validation only | Pass |
| Models | Saved models reproduce saved predictions; metrics recomputed; final model retrained to identical predictions | Pass, 5 of 5 |
| Selection | Saved choice equals the written protocol | Pass |
| Prediction | Every category scored; 15 invalid inputs and a missing field rejected; extreme profiles; contributions add up | Pass |
| App pages | All 7 load headless without exception | Pass |
| App behaviour | Filters, empty result, weighted toggle, preprocessing controls, 4 algorithms, 4 strategies, thresholds, importance views, fairness views, presets, sweep of all 26 inputs | Pass |
| Consistency | App shows the saved test metrics; predictor equals the model's own probability | Pass |
| Browser check | Overview, Consumer analytics, BNPL predictor and Consumer simulator viewed; live prediction made | Pass, layout fixes made |
| Public deployment | https://bnpl-intelligence.streamlit.app opened without signing in: all 7 pages load; predictor with starting values gives 4.0% ("Unlikely BNPL user"); ML laboratory trained a Logistic Regression in 0.7 s | Pass |
| Narrow screens | Overview at 375 px, 1024 px and 1440 px after the layout fix | Pass, no sideways scrolling |
| Documents | Report rendered through Word; figures, contents, page numbers checked; personal metadata removed | Pass |

## Errors found and fixed
Dollar signs shown as formulas; over-claiming figure title; strategy name "None" read as missing; local path in `requirements.txt`; simulator control order; chart overflow; wrong figure references; an overstated claim about feature importance; student ID in the repository copy of the notebook; personal name in Word metadata; KPI cards that broke words on narrow screens and a menu that covered the page on phones (found on the public link).

## Remaining limitations
- The free host puts unused apps to sleep; open the link before marking.
- **Docker image not built** (Docker is not installed here).
- Tested on Windows only; tested with Chromium only; the public app itself runs on Linux; not tested with a screen reader or on a physical phone (only an emulated phone width).
- Random Forest and SHAP could differ in the last digits on another CPU or library version; the lockfile limits this.
- One train/validation/test split; bootstrap intervals do not include the variability of retraining.
- Figures in notebook text were checked by hand against the outputs; they will not update by themselves if the data change.
