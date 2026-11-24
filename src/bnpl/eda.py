"""Exploratory statistics: rates with intervals, association measures suited to each variable type."""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg

T = cfg.TARGET
# Published survey-weighted BNPL rates, Board of Governors (2026), Report on the Economic
# Well-Being of U.S. Households in 2025, Table 41. Used only to validate our processing.
FED_TABLE41 = {"All adults": 16, "White": 12, "Black": 29, "Hispanic": 26, "Asian": 11, "Male": 14, "Female": 19,
               "18-29": 22, "30-44": 21, "45-59": 17, "60+": 9, "Under $25k": 18, "$25k-$50k": 23,
               "$50k-$100k": 18, "$100k+": 12}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_table(df, by, order=None):
    """Sample size, unweighted BNPL rate (95% Wilson interval) and survey-weighted rate per group."""
    rows = []
    for g, d in df.groupby(by, observed=True):
        lo, hi = wilson(d[T].sum(), len(d))
        rows.append({by: g, "n": len(d), "Unweighted rate": d[T].mean(), "CI low": lo, "CI high": hi,
                     "Weighted rate": np.average(d[T], weights=d["weight"])})
    t = pd.DataFrame(rows)
    if order is not None and len(t):
        t[by] = pd.Categorical(t[by], [o for o in order if o in set(t[by])], ordered=True)
        t = t.sort_values(by).reset_index(drop=True)
    return t


def fed_validation(df):
    from .experiments import add_groups
    d = add_groups(df)
    rows = [("All adults", np.average(d[T], weights=d["weight"]), len(d))]
    for col in ["race", "gender", "Age group", "Income group"]:
        for g, s in d.groupby(col):
            rows.append((g, np.average(s[T], weights=s["weight"]), len(s)))
    t = pd.DataFrame(rows, columns=["Group", "Our weighted rate (%)", "n"])
    t = t[t.Group.isin(FED_TABLE41)]
    t["Our weighted rate (%)"] = (t["Our weighted rate (%)"] * 100).round(1)
    t["Fed published (%)"] = t.Group.map(FED_TABLE41)
    t["Difference"] = (t["Our weighted rate (%)"] - t["Fed published (%)"]).round(1)
    return t.reset_index(drop=True)


def codes(df):
    """Numeric and ordinal features as ranks (ordinal order comes from the codebook)."""
    out = df[cfg.NUMERIC].astype(float).copy()
    for c, cats in cfg.ORDINALS.items():
        out[c] = df[c].map({v: i for i, v in enumerate(cats)}).astype(float)
    return out


def spearman_matrix(df):
    """Spearman rank correlation: valid for numeric and ordered categories, unlike Pearson on codes."""
    m = codes(df)
    m["BNPL use"] = df[T]
    return m.corr(method="spearman")


def cramers_v(a, b):
    """Bias-corrected Cramer's V (0 = no association, 1 = perfect)."""
    tab = pd.crosstab(a, b)
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    n, (r, k) = tab.to_numpy().sum(), tab.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-9)), p


def association(df):
    """One comparable table: Cramer's V for every categorical/ordinal feature; rank-biserial for numeric."""
    rows = []
    feats = cfg.FEATURES + cfg.PROTECTED
    for f in feats:
        if f in cfg.NUMERIC:
            u, p = stats.mannwhitneyu(df.loc[df[T] == 1, f], df.loc[df[T] == 0, f])
            n1, n0 = (df[T] == 1).sum(), (df[T] == 0).sum()
            rows.append((f, "Rank-biserial (Mann-Whitney)", 2 * u / (n1 * n0) - 1, p))
        else:
            v, p = cramers_v(df[f].fillna("Not reported"), df[T])
            rows.append((f, "Cramer's V (chi-square)", v, p))
    t = pd.DataFrame(rows, columns=["Feature", "Measure", "Strength", "p-value"])
    m = len(t)                                                   # Holm correction: many tests at once
    o = t["p-value"].rank(method="first")
    t = t.assign(_o=o).sort_values("_o")
    t["Holm-adjusted p"] = np.minimum(1, np.maximum.accumulate(t["p-value"] * (m - t["_o"] + 1)))
    t.insert(1, "Label", t.Feature.map(cfg.LABELS))
    t["Abs"] = t.Strength.abs()
    return t.sort_values("Abs", ascending=False).drop(columns=["_o", "Abs"]).reset_index(drop=True)


def descriptive(df):
    d = df[cfg.NUMERIC].astype(float)
    t = d.describe().T
    t["variance"] = d.var()
    t["skew"] = d.skew()
    t["median"] = d.median()
    t["mode"] = d.mode().iloc[0]
    q1, q3 = d.quantile(.25), d.quantile(.75)
    t["IQR outliers"] = ((d < q1 - 1.5 * (q3 - q1)) | (d > q3 + 1.5 * (q3 - q1))).sum()
    t.index = [cfg.LABELS[i] for i in t.index]
    return t[["count", "mean", "median", "mode", "std", "variance", "min", "25%", "75%", "max", "skew", "IQR outliers"]]


def vif(df):
    """Variance inflation factors for numeric + ordinal inputs (multicollinearity check for LR)."""
    x = codes(df).dropna()
    z = (x - x.mean()) / x.std()
    inv = np.linalg.inv(np.corrcoef(z.to_numpy().T))
    return pd.DataFrame({"Feature": x.columns, "Label": [cfg.LABELS[c] for c in x.columns],
                         "VIF": np.diag(inv)}).sort_values("VIF", ascending=False).reset_index(drop=True)
