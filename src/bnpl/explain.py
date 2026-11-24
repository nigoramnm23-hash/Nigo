"""Predict and explain one profile with the saved pipeline (used by the app and by the tests)."""
import numpy as np
import pandas as pd

from . import config as cfg
from . import pipeline as pl

CATEGORIES = {
    **cfg.ORDINALS,
    "employment": ["Working", "Retired", "Not working"],
    "partner_status": ["Married", "Living with partner", "Never married", "Divorced, separated or widowed"],
    "housing": ["Owner with mortgage", "Owner, no mortgage", "Renter", "Neither owns nor rents"],
    "region": ["Northeast", "Midwest", "South", "West"],
    "metro": ["Metro", "Non-Metro"],
    "card_behaviour": ["No credit card", "Pays in full", "Carried balance once", "Carries balance sometimes", "Carries balance mostly"],
    "credit_confidence": ["Very confident", "Somewhat confident", "Not confident", "Don't know"],
    "credit_score": ["Excellent", "Good", "Fair", "Poor", "Very poor", "Don't know", "Not reported"],
    "spending_vs_income": ["Less than your income", "The same as your income", "More than your income"],
    "bills_status": ["Paid in full without difficulty", "Paid in full with difficulty", "Not all paid in full"],
}
RANGES = {"age": (18, 98), "household_size": (1, 12), "children_u18": (0, 9), "afs_count": (0, 5), "medical_skips": (0, 5)}


def validate(profile):
    """Return a list of human-readable problems; an empty list means the profile can be scored."""
    errors = []
    for f in cfg.FEATURES:
        if f not in profile or profile[f] is None:
            errors.append(f"{cfg.LABELS[f]}: value is missing.")
    if errors:
        return errors
    for f, (lo, hi) in RANGES.items():
        v = profile[f]
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
            errors.append(f"{cfg.LABELS[f]}: must be a number.")
        elif not lo <= v <= hi or float(v) != int(v):
            errors.append(f"{cfg.LABELS[f]}: must be a whole number from {lo} to {hi}.")
    for f, cats in CATEGORIES.items():
        if profile[f] not in cats:
            errors.append(f"{cfg.LABELS[f]}: '{profile[f]}' is not an allowed category.")
    if not errors:
        partner = 1 if profile["partner_status"] == "Living with partner" else 0
        if profile["household_size"] < 1 + partner + profile["children_u18"]:
            errors.append("Household size is too small for the partner and children entered.")
        if profile["employment"] == "Retired" and profile["age"] < 35:
            errors.append("'Retired' below age 35 was treated as implausible in the training data; choose another status.")
        if profile["education"] == cfg.EDUCATION[-1] and profile["age"] < 19:
            errors.append("A bachelor's degree below age 19 is outside the plausible range.")
    return errors


def frame(profile):
    d = {f: profile[f] for f in cfg.FEATURES}
    if d["credit_score"] == "Not reported":
        d["credit_score"] = np.nan           # same path as a real non-response: imputed inside the pipeline
    return pd.DataFrame([d])[cfg.FEATURES]


def predict(pipe, profile):
    problems = validate(profile)
    if problems:
        raise ValueError("; ".join(problems))
    return float(pipe.predict_proba(frame(profile))[0, 1])


def background(pipe, X):
    """Average transformed row: the reference point for contributions."""
    return np.asarray(pipe[:-1].transform(X)).mean(axis=0)


def contributions(pipe, profile, bg, tree_explainer=None):
    """Per-feature contribution in log-odds relative to the average respondent.

    Logistic Regression: exact, coefficient x (value - average). Tree ensembles: TreeSHAP.
    Returns (table, base) where base + contributions.sum() equals the model's log-odds.
    """
    _, src = pl.transformed_names(pipe)
    x = np.asarray(pipe[:-1].transform(frame(profile)))
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        phi = clf.coef_[0] * (x[0] - bg)
        base = float(clf.intercept_[0] + clf.coef_[0] @ bg)
    else:
        sv = tree_explainer.shap_values(x)
        sv = sv[..., 1] if np.ndim(sv) == 3 else sv
        phi, base = sv[0], float(np.ravel(tree_explainer.expected_value)[-1])
    t = pd.DataFrame({"Feature": src, "phi": phi}).groupby("Feature", as_index=False)["phi"].sum()
    t["Label"] = t.Feature.map(cfg.LABELS)
    t["Value"] = [str(profile.get(f, "derived")) for f in t.Feature]
    return t.reindex(t.phi.abs().sort_values(ascending=False).index).reset_index(drop=True), base
