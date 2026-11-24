"""Preprocessing and model pipelines. Everything learned from data is fitted inside these pipelines."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as cfg

STRATEGIES = ["None", "Class weights", "Random oversampling", "SMOTENC"]
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived features; learn only the winsorising limit for household size (training rows)."""

    def __init__(self, derive=True):
        self.derive = derive

    def fit(self, X, y=None):
        self.hh_cap_ = float(np.nanpercentile(X["household_size"], 99))
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        raw_hh = X["household_size"].astype(float)
        X["household_size"] = raw_hh.clip(upper=self.hh_cap_)
        if self.derive:
            # Square-root equivalence scale: the same income stretches further in a smaller household.
            mid = X["income_band"].map(cfg.INCOME_MIDPOINT).astype(float)
            X["log_equiv_income"] = np.log1p(mid / np.sqrt(raw_hh.clip(lower=1)))
            X["child_share"] = (X["children_u18"].astype(float) / raw_hh.clip(lower=1)).clip(upper=1)
            X["fragility_score"] = (
                X["financial_wellbeing"].isin(cfg.WELLBEING[:2]).astype(int)
                + X["emergency_fund"].eq("No").astype(int)
                + X["can_cover_400"].eq("No").astype(int)
                + X["bills_status"].ne("Paid in full without difficulty").astype(int)
                + X["spending_vs_income"].eq("More than your income").astype(int)
                + (X["medical_skips"].astype(float) > 0).astype(int))
        for c in X.columns:
            if c in cfg.ORDINALS or c in cfg.NOMINAL or c in cfg.PROTECTED:
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        return X


def _columns(features, derive):
    num = [c for c in cfg.NUMERIC if c in features]
    if derive:
        num = num + cfg.DERIVED
    ords = [c for c in cfg.ORDINALS if c in features]
    nom = [c for c in cfg.NOMINAL + cfg.PROTECTED if c in features]
    return num, ords, nom


def preprocessor(features=None, derive=True, scale=True, credit_missing="category", nominal_as_int=False):
    """ColumnTransformer: impute -> encode -> scale, by variable type."""
    features = features or cfg.FEATURES
    num, ords, nom = _columns(features, derive)
    scaler = [("scale", StandardScaler())] if scale else []
    nom_enc = (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) if nominal_as_int
               else OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    if credit_missing == "category":
        nom_imp = SimpleImputer(strategy="constant", fill_value="Not reported")
    else:
        nom_imp = SimpleImputer(strategy="most_frequent")
    parts = []
    if num:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median"))] + scaler), num))
    if ords:
        parts.append(("ord", Pipeline(
            [("impute", SimpleImputer(strategy="most_frequent")),
             ("encode", OrdinalEncoder(categories=[cfg.ORDINALS[c] for c in ords],
                                       handle_unknown="use_encoded_value", unknown_value=np.nan)),
             ("impute_unknown", SimpleImputer(strategy="most_frequent"))] + scaler), ords))
    if nom:
        parts.append(("nom", Pipeline([("impute", nom_imp), ("encode", nom_enc)]), nom))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def estimator(name, class_weight=None, **params):
    if name == "Logistic Regression":
        kw = dict(C=0.1, l1_ratio=0.0, solver="saga", max_iter=3000, tol=1e-3)
        return LogisticRegression(**{**kw, **params}, class_weight=class_weight, random_state=cfg.SEED)
    if name == "Decision Tree":
        return DecisionTreeClassifier(**params, class_weight=class_weight, random_state=cfg.SEED)
    if name == "Random Forest":
        kw = dict(n_estimators=300, n_jobs=-1)
        return RandomForestClassifier(**{**kw, **params}, class_weight=class_weight, random_state=cfg.SEED)
    if name == "Gradient Boosting":
        kw = dict(early_stopping=False)
        return HistGradientBoostingClassifier(**{**kw, **params}, class_weight=class_weight, random_state=cfg.SEED)
    if name == "Dummy":
        return DummyClassifier(strategy="prior")
    raise ValueError(name)


def build(name, params=None, strategy="None", features=None, derive=True, scale=True,
          credit_missing="category", sampler_kwargs=None):
    """Full pipeline: feature engineering -> preprocessing -> (resampling) -> classifier.

    The imbalanced-learn Pipeline applies resampling during fit only, so validation and test rows
    are never resampled.
    """
    params = params or {}
    features = features or cfg.FEATURES
    fe = ("engineer", FeatureEngineer(derive=derive))
    cw = "balanced" if strategy == "Class weights" else None
    clf = ("clf", estimator(name, class_weight=cw, **params))
    if strategy in ("None", "Class weights"):
        return Pipeline([fe, ("prep", preprocessor(features, derive, scale, credit_missing)), clf])
    if strategy == "Random oversampling":
        return ImbPipeline([fe, ("prep", preprocessor(features, derive, scale, credit_missing)),
                            ("sample", RandomOverSampler(random_state=cfg.SEED)), clf])
    if strategy == "SMOTENC":
        num, ords, nom = _columns(features, derive)
        n_cont = len(num) + len(ords)
        cat_idx = list(range(n_cont, n_cont + len(nom)))
        onehot = ColumnTransformer([("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx)],
                                   remainder="passthrough")
        return ImbPipeline([fe, ("prep", preprocessor(features, derive, True, credit_missing, nominal_as_int=True)),
                            ("sample", SMOTENC(categorical_features=cat_idx, random_state=cfg.SEED,
                                               **(sampler_kwargs or {}))),
                            ("onehot", onehot), clf])
    raise ValueError(strategy)


def split(df):
    """60/20/20 stratified train / validation / test (the 80/20 cut separates the test set first)."""
    dev, test = train_test_split(df, test_size=0.20, stratify=df[cfg.TARGET], random_state=cfg.SEED)
    train, valid = train_test_split(dev, test_size=0.25, stratify=dev[cfg.TARGET], random_state=cfg.SEED)
    return train, valid, test


def xy(df, features=None):
    return df[features or cfg.FEATURES], df[cfg.TARGET].to_numpy()


def transformed_names(pipe):
    """Readable names of the columns seen by the classifier, and the source feature of each."""
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    bases = sorted(cfg.NOMINAL + cfg.PROTECTED, key=len, reverse=True)
    source = []
    for n in names:
        source.append(next((b for b in bases if n.startswith(b + "_") and n != b), n)
                      if n not in cfg.NUMERIC + cfg.DERIVED + list(cfg.ORDINALS) else n)
    return names, source
