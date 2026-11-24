# Development log

All work below was done on **24 November 2026** in one AI-assisted session. No commit was backdated. Further entries should be added by the student as review work happens.

| Stage | What happened | Decision or lesson |
|---|---|---|
| Specification | Read the 11-page brief and the rubric (35 + 65 marks) | The brief has no AI-use clause; the student must check university rules |
| Dataset mismatch | The first ZIP supplied was SHED **2023**; the registered topic is SHED **2025** | Approved by the student: official 2025 file downloaded; 2023 copy removed from the project |
| Codebook | Extracted the 119-page 2025 codebook | `BNPL1` confirmed as the target (10,930 No, 2,004 Yes); one codebook entry (`malefemale`) is misprinted |
| Leakage review | 20 BNPL follow-ups are asked only of users | Excluded, plus credit applications and overdraft fees (construct overlap) |
| Cleaning | 31 imputed targets dropped; profile/survey conflicts resolved in favour of the live survey answer | Every rule is logged in `outputs/tables/cleaning_log.csv` |
| Validation | Weighted rates compared with the Fed's published Table 41 | All 15 groups within 0.6 points |
| Protocol | Selection rules written before modelling (`experiments.PROTOCOL`) | Avoids choosing rules to suit results |
| Modelling | 4 tuned models + dummy, 4 imbalance strategies | Balancing did not help; Logistic Regression selected by the one-standard-error rule |
| Bug: dollar signs | `$5,000 to $9,999` rendered as a formula in matplotlib, and would in Streamlit text | Math parsing off; dollar signs escaped |
| Bug: figure title | Figure 9 title over-claimed and plotted the first age band at 15 | Neutral title, correct age bands |
| Bug: "None" | Strategy name "None" was read from CSV as a missing value (found by the tests) | Renamed "No balancing"; outputs regenerated |
| Bug: local path | `requirements.txt` header contained a local Windows path | Comments stripped |
| Deployment | Repository pushed, app deployed on Streamlit Community Cloud with the student's accounts | All pages checked without signing in |
| Bug: narrow screens | KPI cards broke words and the menu covered the page on phones (found on the public link) | Wrapping card grid; menu closed by default on phones |
| App | Seven pages, tested headless and in a browser | Dev server bound to localhost only |
