import numpy as np
import pandas as pd

from src.structure import add_structure

DAYS_EMPLOYED_SENTINEL = 365243
NA_FLAG_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza basada en hallazgos del EDA (notebooks/01_eda.ipynb)."""
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df.loc[df["DAYS_EMPLOYED_ANOM"] == 1, "DAYS_EMPLOYED"] = np.nan
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("XNA", np.nan)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features v1 (adoptadas desde exp/optuna-v4, CV 0.76145)."""
    df = df.copy()
    # flags de nulos: el faltante predice (ver EDA)
    for c in NA_FLAG_COLS:
        df[f"{c}_ISNA"] = df[c].isna().astype(int)
    # ratios financieros
    df["CREDIT_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["ANNUITY_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["DOWNPAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["INCOME_PER_FAM"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    # tiempo
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_BIRTH_PCT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
    df["PHONE_CHANGE_YEARS"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25
    # buro y contacto
    bur = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    df["BUREAU_REQ_TOTAL"] = df[bur].sum(axis=1, min_count=1)
    df["PHONE_FLAGS_SUM"] = df[["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE"]].sum(axis=1)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]
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
    # señal estructural sin target (exp v34): necesita train+test juntos (grupos y frecuencias)
    st = add_structure(pd.concat([train.drop(columns="TARGET", errors="ignore"), test], ignore_index=True))
    train = pd.concat([add_features(clean(train)), st.iloc[:len(train)].set_axis(train.index)], axis=1)
    test = pd.concat([add_features(clean(test)), st.iloc[len(train):].set_axis(test.index)], axis=1)
    return to_category(train, test)
