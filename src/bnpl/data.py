"""Load SHED 2025 and turn raw survey columns into clean, documented model features.

Everything here is row-wise and stateless: no value is learned from other rows, so it cannot
leak information between training and test data. Anything learned from data (imputation values,
scaling, encoding, winsorising limits) lives in pipeline.py and is fitted on training rows only.
"""
import hashlib
import zipfile

import numpy as np
import pandas as pd

from . import config as cfg

RAW_COLS = [
    "shedid", "weight", "BNPL1", "BNPL1_iflag", "ppage", "pphhsize", "ppkid017", "ppt18ov", "I40",
    "educ_4cat", "ED0", "ppemploy", "D1A", "D1I", "ppmarit5", "L0_a", "L0_b", "GH1", "ppreg4", "ppmsacat",
    "B2", "B3", "EF1", "pay_casheqv", "EF5C", "EF5D", "C2A", "C4A", "A6", "I20", "I9", "FD3", "SL21",
    "K21_a", "BK1", "BK2_a", "BK2_b", "BK2_c", "BK2_d", "BK2_e",
    "E1_a", "E1_b", "E1_c", "E1_d", "E1_e", "ppfs1482", "malefemale", "race_5cat",
]
# Follow-ups asked only of BNPL users: they reveal the target, so they can never be predictors.
BNPL_FOLLOWUPS = ["BNPL3", "BNPL3A", "BNPL1A"] + [f"BNPL4_{c}" for c in "abcdefg"] + ["BNPL5"] + \
                 [f"BNPL6_{c}" for c in "abcdefg"] + ["BNPL7_a", "BNPL7_b"]


def load_raw(columns=None):
    """Read the original CSV straight out of the official ZIP (UTF-8)."""
    with zipfile.ZipFile(cfg.RAW_ZIP) as z, z.open(cfg.RAW_MEMBER) as f:
        return pd.read_csv(f, usecols=columns, low_memory=False, encoding="utf-8")


def raw_sha256():
    return hashlib.sha256(cfg.RAW_ZIP.read_bytes()).hexdigest()


def _yes(s):
    return s.eq("Yes")


def _tidy(s):
    """Normalise typographic quotes/dashes and stray spaces so category labels are consistent."""
    return s.str.replace("’", "'", regex=False).str.replace("–", "-", regex=False).str.strip()


def clean(raw):
    """Return (features + audit columns + target, audit log of every correction)."""
    log = []
    df = raw.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = _tidy(df[c])

    n0 = len(df)
    df = df[~df["shedid"].duplicated()]
    log.append(("Duplicate respondent IDs removed", n0 - len(df)))
    imputed = df["BNPL1_iflag"].eq(1)
    df = df[~imputed].reset_index(drop=True)
    log.append(("Rows dropped: target answer was imputed by the survey, not reported", int(imputed.sum())))

    out = pd.DataFrame({"shedid": df["shedid"].astype(str), "weight": df["weight"]})
    out[cfg.TARGET] = _yes(df["BNPL1"]).astype(int)

    # Household: the live survey answer (Oct 2025) overrides the older panel profile on conflict.
    children = np.maximum(df["ppkid017"], _yes(df["L0_b"]).astype(int))
    minimum = 1 + _yes(df["L0_a"]).astype(int) + children
    log.append(("Children corrected to at least 1 (survey says own child at home, profile says 0)",
                int((children > df["ppkid017"]).sum())))
    log.append(("Household size raised to the minimum implied by partner and children",
                int((minimum > df["pphhsize"]).sum())))
    out["age"] = df["ppage"]
    out["household_size"] = np.maximum(df["pphhsize"], minimum)
    out["children_u18"] = children

    out["afs_count"] = sum(_yes(df[f"BK2_{c}"]).astype(int) for c in "abcde")
    out["medical_skips"] = sum(_yes(df[f"E1_{c}"]).astype(int) for c in "abcde")

    out["income_band"] = df["I40"]
    edu = df["educ_4cat"].copy()
    bad = (df["ppage"] < 22) & df["ED0"].str.contains("Master|Doctor|Professional", na=False)
    edu[bad] = np.nan
    log.append(("Education set to missing: graduate degree reported below age 22 (implausible)", int(bad.sum())))
    out["education"] = edu
    out["financial_wellbeing"] = df["B2"]
    out["change_vs_last_year"] = df["B3"]
    out["food_security"] = df["FD3"]
    out["income_volatility"] = df["I9"]
    out["emergency_fund"] = df["EF1"]
    out["can_cover_400"] = df["pay_casheqv"]
    out["student_loan"] = df["SL21"]
    out["retirement_account"] = df["K21_a"]
    out["bank_account"] = df["BK1"]

    retired = _yes(df["D1I"])
    young = retired & (df["ppage"] < 35) & ~_yes(df["D1A"])
    log.append(("'Retired' under age 35 recoded to 'Not working' (implausible)", int(young.sum())))
    out["employment"] = np.select(
        [_yes(df["D1A"]), retired & (df["ppage"] >= 35)], ["Working", "Retired"], "Not working")
    out["partner_status"] = np.select(
        [df["ppmarit5"].eq("Now married"), _yes(df["L0_a"]), df["ppmarit5"].eq("Never married")],
        ["Married", "Living with partner", "Never married"], "Divorced, separated or widowed")
    out["housing"] = df["GH1"].map({
        "Own your home with a mortgage or loan": "Owner with mortgage",
        "Own your home free and clear (without a mortgage or loan)": "Owner, no mortgage",
        "Pay rent": "Renter", "Neither own nor pay rent": "Neither owns nor rents"})
    out["region"] = df["ppreg4"]
    out["metro"] = df["ppmsacat"]
    # Structural missingness: C4A is blank only because the person has no card -> its own category.
    out["card_behaviour"] = df["C4A"].map({
        "Never carried an unpaid balance (always pay in full)": "Pays in full", "Once": "Carried balance once",
        "Some of the time": "Carries balance sometimes", "Most or all of the time": "Carries balance mostly"}
    ).where(_yes(df["C2A"]), "No credit card")
    out["credit_confidence"] = df["A6"]
    out["credit_score"] = df["ppfs1482"]           # genuinely missing for about 23% -> handled in pipeline
    out["spending_vs_income"] = df["I20"]
    # EF5D is blank only when EF5C == No -> combine the two questions into one variable.
    out["bills_status"] = np.select([df["EF5C"].eq("No"), df["EF5D"].eq("Yes")],
                                    ["Not all paid in full", "Paid in full with difficulty"],
                                    "Paid in full without difficulty")
    out["gender"] = df["malefemale"]
    out["race"] = df["race_5cat"]
    return out, pd.DataFrame(log, columns=["Check", "Rows affected"])


def load_clean(refresh=False):
    """Cleaned analysis table (cached as CSV for the app)."""
    path = cfg.PROCESSED / "shed_bnpl_clean.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"shedid": str}, keep_default_na=True)
    out, log = clean(load_raw(RAW_COLS))
    out.to_csv(path, index=False, encoding="utf-8")
    log.to_csv(cfg.TABLES / "cleaning_log.csv", index=False)
    return pd.read_csv(path, dtype={"shedid": str})


# name: (source variables, question/definition from the codebook, type, missing treatment, why)
_D = {
    "bnpl_user": ("BNPL1", "In the past year, used a 'Buy Now, Pay Later' service to buy something (Yes=1, No=0)", "Target (binary)", "31 imputed answers dropped", "TARGET"),
    "age": ("ppage", "Age in years", "Numeric", "None missing; median imputer as safeguard", "BNPL is a retail-credit product with a known age gradient"),
    "household_size": ("pphhsize, L0_a, L0_b", "People in the household, raised to the minimum implied by partner/children", "Numeric", "None; winsorised at training 99th percentile", "Spending needs; needed to equivalise income"),
    "children_u18": ("ppkid017, L0_b", "Children aged 0-17 in the household", "Numeric", "No missing values", "Family spending pressure"),
    "afs_count": ("BK2_a to BK2_e", "Count of: non-bank money order, non-bank check cashing, payday loan, pawn/auto-title loan, tax refund advance (past 12 months)", "Numeric 0-5", "No missing values", "Use of alternatives to mainstream credit"),
    "medical_skips": ("E1_a to E1_e", "Count of care types gone without because of cost (past 12 months)", "Numeric 0-5", "No missing values", "Material hardship indicator"),
    "income_band": ("I40", "Respondent and spouse/partner total income, past 12 months (15 bands)", "Ordinal", "None; mode safeguard", "Core affordability variable"),
    "education": ("educ_4cat, ED0", "Education, 4 categories", "Ordinal", "Implausible values set missing, then most frequent value", "Socio-economic position"),
    "financial_wellbeing": ("B2", "How well managing financially these days", "Ordinal", "No missing values", "Headline SHED well-being measure"),
    "change_vs_last_year": ("B3", "Better or worse off than 12 months ago", "Ordinal", "No missing values", "Direction of finances"),
    "food_security": ("FD3", "Food eaten in the household in the past month", "Ordinal", "No missing values", "Material hardship indicator"),
    "income_volatility": ("I9", "Month-to-month income stability", "Ordinal", "No missing values", "Cash-flow smoothing is a proposed BNPL motive"),
    "emergency_fund": ("EF1", "Has three months of emergency savings", "Binary", "No missing values", "Resilience"),
    "can_cover_400": ("pay_casheqv", "Would cover a $400 expense with cash or equivalent", "Binary", "No missing values", "Resilience"),
    "student_loan": ("SL21", "Has any student loans", "Binary", "No missing values", "Existing debt commitment"),
    "retirement_account": ("K21_a", "Has a retirement savings account", "Binary", "No missing values", "Asset holding"),
    "bank_account": ("BK1", "Has a checking, savings or money market account", "Binary", "No missing values", "Access to payments"),
    "employment": ("D1A, D1I, ppage", "Working / Retired / Not working (live survey answers)", "Nominal", "No missing values", "Income source; replaces stale profile variable ppemploy"),
    "partner_status": ("ppmarit5, L0_a", "Married / Living with partner / Never married / Previously married", "Nominal", "No missing values", "Marital status"),
    "housing": ("GH1", "Housing tenure", "Nominal", "No missing values", "Wealth and fixed costs"),
    "region": ("ppreg4", "Census region", "Nominal", "No missing values", "Geographic control"),
    "metro": ("ppmsacat", "Metropolitan area status", "Nominal", "No missing values", "Geographic control"),
    "card_behaviour": ("C2A, C4A", "Has a card, and how often an unpaid balance was carried (12 months)", "Nominal", "Structural blanks become 'No credit card'", "Closest substitute product"),
    "credit_confidence": ("A6", "Confidence that a credit card application would be approved", "Nominal", "'Don't know' kept as a category", "Perceived credit access"),
    "credit_score": ("ppfs1482", "Self-rated credit score (panel profile)", "Nominal", "23% missing become 'Not reported' (tested against mode imputation)", "Perceived credit quality"),
    "spending_vs_income": ("I20", "Spending compared with income last month", "Nominal", "No missing values", "Budget pressure"),
    "bills_status": ("EF5C, EF5D", "Paid all non-card bills in full last month, and with or without difficulty", "Nominal", "Structural blanks combined into one variable", "Payment difficulty"),
    "gender": ("malefemale", "Sex at birth where available, otherwise gender", "Audit only", "No missing values", "EXCLUDED as input (protected); used for fairness audit"),
    "race": ("race_5cat", "Race/ethnicity, 5 categories", "Audit only", "No missing values", "EXCLUDED as input (protected); used for fairness audit"),
}
EXCLUDED = [
    ("BNPL1A, BNPL3, BNPL3A, BNPL4_a-g, BNPL5, BNPL6_a-g, BNPL7_a-b", "Target leakage: asked only of BNPL users, so being non-blank reveals the answer"),
    ("BNPL1_iflag and all *_iflag", "Survey imputation flags, not characteristics of the person"),
    ("A0, A7_a-f, A1_a-c, A8, A2, A0B", "Construct overlap: applying for 'other credit or loan' in the same 12 months may itself be a BNPL application"),
    ("BK2_f (overdraft fee)", "Possible consequence of BNPL: BNPL1A shows BNPL payments can trigger overdrafts"),
    ("EF8_h, EF3_f", "Borrowing measures that may overlap with BNPL use"),
    ("EF7, EF7A", "Split-sample question: each half of the sample saw a different answer scale"),
    ("ppfsasset", "34% missing and overlaps with EF1 and pay_casheqv"),
    ("ppemploy, ppinc7, ppeducat", "Older panel-profile versions; the live survey answer is used instead"),
    ("ppage groups, inc_4cat_50k, pprent, atleast_okay", "Duplicates of selected variables"),
    ("gender, race, pppa_lgb, pphispan", "Protected characteristics: excluded on ethical grounds; gender and race kept for audit"),
    ("weight*, panel_weight*", "Design weights: used for population estimates, not as predictors"),
    ("shedid, duration, field_*, *date", "Identifiers and survey-process data"),
    ("pph* health items, ppp*, pphi*", "Sensitive health/political items with no clear business need"),
]


def feature_dictionary(raw, cleaned):
    rows = []
    for name, (src, desc, typ, miss, why) in _D.items():
        first = src.split(",")[0].split(" to ")[0].strip()
        cats = ""
        if first in raw.columns and raw[first].nunique() <= 15:
            cats = " | ".join(str(v) for v in raw[first].dropna().unique())
        rows.append({"Feature": name, "Human-readable name": cfg.LABELS.get(name, "BNPL user (target)"),
                     "Source variable(s)": src, "Description": desc, "Type": typ,
                     "Original response categories (first source)": cats or "numeric",
                     "Missing after cleaning (%)": round(cleaned[name].isna().mean() * 100, 2),
                     "Missing-value treatment": miss, "Reason": why,
                     "Used as model input": "Yes" if name in cfg.FEATURES else "No"})
    ex = pd.DataFrame(EXCLUDED, columns=["Excluded variable(s)", "Reason for exclusion"])
    return pd.DataFrame(rows), ex
