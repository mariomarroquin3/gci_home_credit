import numpy as np
import pandas as pd

DAYS_EMPLOYED_SENTINEL = 365243


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza basada en hallazgos del EDA (notebooks/01_eda.ipynb)."""
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df.loc[df["DAYS_EMPLOYED_ANOM"] == 1, "DAYS_EMPLOYED"] = np.nan
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("XNA", np.nan)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_MEAN"] = ext.mean(axis=1)
    df["EXT_STD"] = ext.std(axis=1)
    df["EXT_PROD"] = ext.prod(axis=1, min_count=3)
    df["EXT_NA_COUNT"] = ext.isna().sum(axis=1)
    return df


def to_category(train: pd.DataFrame, test: pd.DataFrame):
    """Categorías consistentes entre train y test para LightGBM."""
    cats = train.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in cats:
        levels = pd.Index(train[c].dropna().unique()).union(test[c].dropna().unique())
        train[c] = pd.Categorical(train[c], categories=levels)
        test[c] = pd.Categorical(test[c], categories=levels)
    return train, test, cats


def build(train: pd.DataFrame, test: pd.DataFrame):
    train = add_features(clean(train))
    test = add_features(clean(test))
    return to_category(train, test)
