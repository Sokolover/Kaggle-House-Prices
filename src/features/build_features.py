# 01_eda.ipynb
# ├─ визуализация и исследование распределений
# ├─ поиск выбросов
# ├─ вывод: какие колонки/границы потенциальных выбросов
#
# 02_features.py
# ├─ функции feature engineering
# ├─ функция drop_outliers(train_df) → возвращает train без аномалий
# ├─ остальные фичи
#
# EDA = исследование → не должно менять данные
# Feature Engineering = подготовка данных → здесь уже меняем train
#
import pandas as pd
import numpy as np
from scipy.stats import skew

ohe_numeric_features = ["MSSubClass"]
ordinal_numeric_features = ["OverallQual", "OverallCond"]
cyclic_numeric_features = ["MoSold"]

ordinal_features = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "KitchenQual",
    "GarageQual",
    "GarageCond",
    "FireplaceQu",
    "PoolQC",
    "Functional",
    "GarageFinish",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "HeatingQC",
] + ordinal_numeric_features

ohe_features = [
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "GarageType",
    "PavedDrive",
    "MiscFeature",
    "Fence",
    "SaleType",
    "SaleCondition",
    "Electrical",
]

ohe_with_grouping = [
    "Exterior1st",
    "Exterior2nd",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "RoofMatl",
] + ohe_numeric_features

# ExterQual, ExterCond, BsmtQual, BsmtCond, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC
quality_map = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
# BsmtExposure
bsmt_exposure_map = {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
# GarageFinish
garage_finish_map = {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3}
# BsmtFinType1, BsmtFinType2
bsmt_fin_type_map = {
    "NA": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6,
}
# Functional
functional_map = {
    "NA": 0,
    "Sal": 1,
    "Sev": 2,
    "Maj2": 3,
    "Maj1": 4,
    "Mod": 5,
    "Min2": 6,
    "Min1": 7,
    "Typ": 8,
}

ordinal_maps = {
    "ExterQual": quality_map,
    "ExterCond": quality_map,
    "BsmtQual": quality_map,
    "BsmtCond": quality_map,
    "HeatingQC": quality_map,
    "KitchenQual": quality_map,
    "FireplaceQu": quality_map,
    "GarageQual": quality_map,
    "GarageCond": quality_map,
    "PoolQC": quality_map,
    "BsmtExposure": bsmt_exposure_map,
    "GarageFinish": garage_finish_map,
    "BsmtFinType1": bsmt_fin_type_map,
    "BsmtFinType2": bsmt_fin_type_map,
    "Functional": functional_map,
}


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    show_categorical_features(df)
    show_categorical_features_represented_by_numbers(df)

    df = apply_encoding_as_string_type_for_numeric_categorical_features(df)
    df = apply_ordinal_ecoding(df)
    df = apply_one_hot_encoding(df)

    show_if_all_categorical_features_were_encoded(df)

    new_cyclic_numeric_features = apply_cyclic_numeric_features_encoding(df)
    [df, new_numerical_features] = create_new_numerical_features(df)
    print(df[new_numerical_features].head())

    remain_numeric_cols = remove_redundant_numeric_features(df)

    [df, log_transformed_features] = log_transform_train_numerical_features(
        df, remain_numeric_cols, new_cyclic_numeric_features
    )
    apply_sparse_handling_on_log_transformed_features(df, log_transformed_features)
    return [df, log_transformed_features, new_cyclic_numeric_features]


# public
def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outliers = df[(df["GrLivArea"] > 4500) & (df["SalePrice"] < 300000)].index
    return df.drop(outliers)


# public
def show_categorical_features(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f"\n{col} ({len(unique_vals)} unique values):")
        print(unique_vals)


# public
def show_categorical_features_represented_by_numbers(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numeric_cols:
        if df[col].nunique() < 20:
            print(col, "- categorical?")


def apply_encoding_as_string_type_for_numeric_categorical_features(df: pd.DataFrame):
    df[ohe_numeric_features] = df[ohe_numeric_features].astype(str)
    return df


# public
def apply_ordinal_ecoding(df: pd.DataFrame):
    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def apply_group_rare_categories(df: pd.DataFrame):
    for col in ohe_with_grouping:
        df = group_rare_categories(df, col)
    return df


def group_rare_categories(df, col, min_freq=20):
    freqs = df[col].value_counts()
    rare = freqs[freqs < min_freq].index
    df[col] = df[col].replace(rare, "Other")
    return df


def get_all_ohe_features():
    return ohe_features + ohe_with_grouping


# public
def apply_one_hot_encoding(df: pd.DataFrame, drop_first=True):
    df = apply_group_rare_categories(df)
    df = pd.get_dummies(df, columns=get_all_ohe_features(), drop_first=drop_first)
    return df


# public
def show_if_all_categorical_features_were_encoded(df: pd.DataFrame):
    print("Object columns: ", df.select_dtypes(include="object").columns)

    pd.set_option("display.max_rows", None)  # show all rows
    pd.set_option("display.max_columns", None)  # show all columns
    pd.set_option("display.max_colwidth", None)  # show full column names
    pd.set_option("display.width", None)  # don't wrap lines
    print("Column types: ", df.dtypes)

    bad_cols = df.columns[(df.dtypes == "object") | (df.dtypes == "category")]

    print("Bad columns: ", bad_cols)


# public
def apply_cyclic_numeric_features_encoding(df: pd.DataFrame):
    df["MoSold_sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
    df["MoSold_cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)
    return ["MoSold_sin", "MoSold_cos"]


# public
def create_new_numerical_features(df: pd.DataFrame):
    # total square
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    # total bath amount
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )
    # total porch square
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )
    # age of house on the moment it was sold
    df["AgeAtSale"] = df["YrSold"] - df["YearBuilt"]
    # where house remodeled (repaired) or not
    df["Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    # garage age
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    # if there no harage - fill with 0
    df["GarageAge"] = df["GarageAge"].fillna(0)
    # is the house is new or not
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)

    new_features = [
        "TotalSF",
        "TotalBath",
        "AgeAtSale",
        "Remodeled",
        "GarageAge",
        "IsNew",
        "TotalPorchSF",
    ]
    return [df, new_features]


# public
def remove_redundant_numeric_features(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    features_to_drop = ["YrSold", "MoSold"]
    df.drop(columns=features_to_drop, inplace=True)
    numeric_cols = [x for x in numeric_cols if x not in features_to_drop]
    return numeric_cols


# public
def log_transform_train_numerical_features(
    df: pd.DataFrame, numeric_cols, new_cyclic_numeric_features
) -> pd.DataFrame:
    # transform target
    df["SalePrice"] = np.log1p(df["SalePrice"])

    # exclude original target
    numeric_cols.remove("SalePrice")

    # exclude categorical features that are represented in numeric format
    all_categorical_features = ordinal_features + ohe_features

    # exclude cyclic numerical features
    transformed_features = all_categorical_features + new_cyclic_numeric_features

    # exclude all numericly encoded features from numeric value features in numeric_cols
    numeric_cols = [col for col in numeric_cols if col not in transformed_features]

    # exclude binary features
    numeric_cols = [col for col in numeric_cols if df[col].nunique() > 2]

    # find and exclude featues that semantically not needed to be log-transformed
    semantic_exclude = []

    for col in numeric_cols:
        unique_vals = df[col].nunique()
        max_val = df[col].max()

        # categorical / discrete
        if unique_vals < 20:
            semantic_exclude.append(col)

        # counters
        if max_val <= 10 and unique_vals <= 10:
            semantic_exclude.append(col)

    loggable_numeric_cols = [col for col in numeric_cols if col not in semantic_exclude]

    # collect skewed features
    skewed_features = df[loggable_numeric_cols].apply(lambda x: skew(x.dropna()))
    skewed_features = skewed_features[abs(skewed_features) > 0.75].index.tolist()

    # collect features correlated with target
    correlated_features = []
    for col in loggable_numeric_cols:
        corr_original = df[col].corr(df["SalePrice"])
        corr_log = np.log1p(df[col]).corr(df["SalePrice"])
        if abs(corr_log) > abs(corr_original):
            correlated_features.append(col)

    # unite skewed and correlated features
    log_candidates_features = set(skewed_features) | set(correlated_features)

    # after operations with set log_candidates_features have type set[Hashable | Any]
    # need to cast it back to list[str]
    log_transformed_features: list[str] = list(log_candidates_features)

    # apply log-transform
    for col in log_transformed_features:
        df[col] = np.log1p(df[col])

    return [df, log_transformed_features]


# public
def apply_sparse_handling_on_log_transformed_features(
    df: pd.DataFrame, log_transformed_features
):
    # find sparse features and create binary columns for them
    sparsity = df[log_transformed_features].apply(lambda x: (x == 0).mean())
    sparse_features = sparsity[sparsity > 0.5].index.tolist()  # >50% zeros

    for col in sparse_features:
        df[f"Has_{col}"] = (df[col] > 0).astype(int)
