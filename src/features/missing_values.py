import pandas as pd
from pandas.api.types import (
    is_string_dtype,
    is_object_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
)
import matplotlib.pyplot as plt
import seaborn as sns


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    missing_df = pd.DataFrame(
        {"missing_count": missing, "dtype": df[missing.index].dtypes}
    )

    return missing_df


def plot_missing_values(df: pd.DataFrame, title="Missing values"):
    plt.figure(figsize=(50, 10))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title(title)
    plt.show()


def process_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    original_df = df.copy()
    missing_values_df = get_missing_values(df.copy())

    missing_values_df["missing_value_fill_type"] = [
        get_fill_value(col, missing_values_df.loc[col, "dtype"])
        for col in missing_values_df.index
    ]

    return fill_missing_values(original_df, missing_values_df)


def has_missing_values(df: pd.DataFrame) -> bool:
    return df.isnull().sum().sum() > 0


def get_fill_value(col, dtype):
    # Create a dictionary with default values ​​for columns with gaps.
    # There can be special default values like 'median_by_neighborhood' - they will have specific processing rules in process missing values function.
    # Here we only specify special cases; the rest will be handled automatically (like 0 for int/float values).
    default_fill = {
        "PoolQC": "NA",
        "MiscFeature": "NA",
        "Alley": "NA",
        "Fence": "NA",
        "MasVnrType": "None",
        "FireplaceQu": "NA",
        "LotFrontage": "median_by_neighborhood",
        "GarageType": "NA",
        "GarageFinish": "NA",
        "GarageQual": "NA",
        "GarageCond": "NA",
        "BsmtExposure": "NA",
        "BsmtFinType1": "NA",
        "BsmtFinType2": "NA",
        "BsmtQual": "NA",
        "BsmtCond": "NA",
        "Electrical": "SBrkr",  # Fill with moda value
    }

    if col in default_fill:
        return default_fill[col]
    # elif 'object' in str(dtype):
    elif (
        is_object_dtype(dtype) or is_categorical_dtype(dtype) or is_string_dtype(dtype)
    ):
        return "None"
    # elif 'int' in str(dtype) or 'float' in str(dtype):
    elif is_numeric_dtype(dtype):
        return 0
    else:
        return "UNKNOWN"


def fill_missing_values(original_df, missing_values_df):
    for col, row in missing_values_df.iterrows():
        """
        col → index of the row in missing_values_df (the column name in our train/test df)
        row → the entire row as a Series with all its fields

        col = "LotFrontage"
        row = Series(
            missing_count=259,
            dtype="float64",
            missing_value_fill_type="median_by_neighborhood"
        )
        """

        fill_type = row["missing_value_fill_type"]

        # 0/1 flag that shows if column had missing value
        # original_df[col + '_was_missing'] = original_df[col].isnull().astype(int)

        if fill_type == "median_by_neighborhood":
            # (!) todo: here may be data leakage if i apply it on test. need to investigate
            original_df[col] = original_df.groupby("Neighborhood")[col].transform(
                lambda x: x.fillna(
                    x.median()
                )  # By default, pandas ignores NaNs when calculating .median()
            )
        elif fill_type in {"NA", "None", "SBrkr"}:
            original_df[col] = original_df[col].fillna(fill_type)
        elif fill_type == 0:
            original_df[col] = original_df[col].fillna(0)
        elif fill_type == "UNKNOWN":
            print(f"[WARN] Unknown dtype for column '{col}', filling with 0")
            original_df[col] = original_df[col].fillna(0)
        else:
            print(f"[WARN] Unknown fill_type '{fill_type}' for column '{col}'")

    return original_df
