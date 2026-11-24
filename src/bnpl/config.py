"""Paths, constants and category orders shared by the notebook, scripts, app and tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = ROOT / "data" / "raw" / "SHED_public_use_data_2025_(CSV).zip"
RAW_MEMBER = "public2025.csv"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"
for _p in (PROCESSED, TABLES, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "bnpl_user"

# Integrity check: SHA-256 of the official file downloaded on 24 November 2026.
RAW_SHA256 = "a4ab3f7d042f16d63626b1af9aeb6f0b8a0b39e412fa94b87265bc67fe26de14"

# --- Ordered categories (low -> high), spelled exactly as in the 2025 data file -------------
INCOME_BANDS = [
    "Less than $5,000", "$5,000 to $9,999", "$10,000 to $14,999", "$15,000 to $19,999",
    "$20,000 to $24,999", "$25,000 to $29,999", "$30,000 to $34,999", "$35,000 to $39,999",
    "$40,000 to $49,999", "$50,000 to $59,999", "$60,000 to $74,999", "$75,000 to $99,999",
    "$100,000 to $149,999", "$150,000 to $199,999", "$200,000 or more",
]
# Band midpoints in dollars. The top band is open-ended: 250,000 is an ASSUMPTION (documented).
INCOME_MIDPOINT = dict(zip(INCOME_BANDS, [
    2500, 7500, 12500, 17500, 22500, 27500, 32500, 37500, 45000, 55000, 67500, 87500,
    125000, 175000, 250000]))
EDUCATION = ["Less than a high school degree", "High school degree or GED",
             "Some college/technical or associates degree", "Bachelor's degree or more"]
WELLBEING = ["Finding it difficult to get by", "Just getting by", "Doing okay", "Living comfortably"]
CHANGE = ["Much worse off", "Somewhat worse off", "About the same", "Somewhat better off", "Much better off"]
FOOD = ["Often not enough to eat", "Sometimes not enough to eat",
        "Enough, but not always the kinds of food we wanted to eat",
        "Enough of the kinds of food we wanted to eat"]
VOLATILITY = ["Roughly the same amount each month", "Occasionally varies from month to month",
              "Varies quite often from month to month"]
YESNO = ["No", "Yes"]

ORDINALS = {
    "income_band": INCOME_BANDS, "education": EDUCATION, "financial_wellbeing": WELLBEING,
    "change_vs_last_year": CHANGE, "food_security": FOOD, "income_volatility": VOLATILITY,
    "emergency_fund": YESNO, "can_cover_400": YESNO, "student_loan": YESNO,
    "retirement_account": YESNO, "bank_account": YESNO,
}
NUMERIC = ["age", "household_size", "children_u18", "afs_count", "medical_skips"]
NOMINAL = ["employment", "partner_status", "housing", "region", "metro", "card_behaviour",
           "credit_confidence", "credit_score", "spending_vs_income", "bills_status"]
FEATURES = NUMERIC + list(ORDINALS) + NOMINAL          # the 26 model inputs
DERIVED = ["log_equiv_income", "child_share", "fragility_score"]   # created inside the pipeline
PROTECTED = ["gender", "race"]                         # audit only, never model inputs

GROUPS = {
    "Demographic": ["age", "household_size", "children_u18", "education", "employment",
                    "partner_status", "housing", "region", "metro"],
    "Income": ["income_band", "income_volatility", "spending_vs_income"],
    "Financial well-being": ["financial_wellbeing", "change_vs_last_year", "food_security",
                             "medical_skips", "bills_status"],
    "Resilience": ["emergency_fund", "can_cover_400", "retirement_account"],
    "Credit and banking": ["card_behaviour", "credit_confidence", "credit_score", "student_loan",
                           "bank_account", "afs_count"],
}

LABELS = {
    "age": "Age", "household_size": "Household size", "children_u18": "Children under 18",
    "afs_count": "Alternative financial services used (0-5)",
    "medical_skips": "Types of care skipped due to cost (0-5)",
    "income_band": "Annual income", "education": "Education",
    "financial_wellbeing": "Financial well-being", "change_vs_last_year": "Finances vs last year",
    "food_security": "Food sufficiency", "income_volatility": "Income stability",
    "emergency_fund": "Three-month emergency fund",
    "can_cover_400": "Can cover $400 with cash or equivalent",
    "student_loan": "Has student loan", "retirement_account": "Has retirement account",
    "bank_account": "Has bank account", "employment": "Employment status",
    "partner_status": "Partnership status", "housing": "Housing", "region": "Region",
    "metro": "Metro status", "card_behaviour": "Credit-card behaviour",
    "credit_confidence": "Confidence of card approval", "credit_score": "Self-rated credit score",
    "spending_vs_income": "Spending vs income", "bills_status": "Bill payment last month",
    "log_equiv_income": "Equivalised income (log)", "child_share": "Share of children in household",
    "fragility_score": "Financial fragility score (0-6)",
    "gender": "Gender", "race": "Race/ethnicity",
}

# --- Colours: emerald is the brand; model hues follow a CVD-checked order -------------------
C = {"user": "#0B6E4F", "non": "#A9B4AE", "ink": "#14241E", "muted": "#5B6B64", "grid": "#E6E8E3",
     "surface": "#FFFFFF", "accent": "#EB6834"}
MODEL_COLOURS = {"Logistic Regression": "#2A78D6", "Decision Tree": "#EB6834",
                 "Random Forest": "#1BAF7A", "Gradient Boosting": "#6250D6", "Dummy": "#8A938E"}
MODEL_DASH = {"Logistic Regression": "solid", "Decision Tree": "dash",
              "Random Forest": "dashdot", "Gradient Boosting": "dot", "Dummy": "longdash"}
