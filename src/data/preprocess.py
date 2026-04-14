from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "SeriousDlqin2yrs"

REQUIRED_COLUMNS = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validates that the expected columns exist in the dataset.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def validate_numeric_columns(df: pd.DataFrame) -> None:
    """
    Validates that numeric columns are truly numeric.
    """
    numeric_cols = [col for col in REQUIRED_COLUMNS if col != TARGET_COLUMN]

    non_numeric = [
        col for col in numeric_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise TypeError(f"These columns must be numeric: {non_numeric}")


def validate_no_missing_values(df: pd.DataFrame) -> None:
    """
    Ensures no missing values remain after preprocessing.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(f"Missing values remain after preprocessing:\n{missing}")


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes known unwanted columns such as auto-generated index columns.
    """
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles unrealistic ages by setting them to NaN.
    """
    df = df.copy()
    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan
    return df


def create_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary indicators for missingness before imputation.
    """
    df = df.copy()
    df["MonthlyIncome_was_missing"] = df["MonthlyIncome"].isnull().astype("int64")
    df["NumberOfDependents_was_missing"] = df["NumberOfDependents"].isnull().astype("int64")
    df["age_was_invalid"] = df["age"].isnull().astype("int64")
    return df


def fit_preprocessing_artifacts(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Learns preprocessing values from a dataframe.
    This should be fitted on training data only in a production workflow.
    """
    artifacts = {
        "monthly_income_median": df["MonthlyIncome"].median(),
        "number_of_dependents_fill": 0,
        "age_median": df["age"].median(),
        "rev_upper": df["RevolvingUtilizationOfUnsecuredLines"].quantile(0.99),
        "debt_upper": df["DebtRatio"].quantile(0.99),
        "income_upper": df["MonthlyIncome"].quantile(0.99),
        "delinquency_upper": 20,
        "dependents_upper": 10,
        "real_estate_upper": 10,
        "open_credit_upper": 50,
    }
    return artifacts


def cap_series(
    series: pd.Series,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> pd.Series:
    """
    Caps a pandas Series using fixed lower/upper bounds.
    """
    return series.clip(lower=lower, upper=upper)


def transform_with_artifacts(df: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies preprocessing transformations using fitted artifacts.
    """
    df = df.copy()

    # Invalid ages handled before imputing
    df = clean_age(df)

    # Missing value imputation
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(artifacts["monthly_income_median"])
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        artifacts["number_of_dependents_fill"]
    )
    df["age"] = df["age"].fillna(artifacts["age_median"])

    # Outlier treatment
    df["RevolvingUtilizationOfUnsecuredLines"] = cap_series(
        df["RevolvingUtilizationOfUnsecuredLines"],
        lower=0,
        upper=artifacts["rev_upper"],
    )

    df["DebtRatio"] = cap_series(
        df["DebtRatio"],
        lower=0,
        upper=artifacts["debt_upper"],
    )

    df["MonthlyIncome"] = cap_series(
        df["MonthlyIncome"],
        lower=0,
        upper=artifacts["income_upper"],
    )

    # Delinquency counts above 20 are considered unrealistic in this dataset
    delinquency_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]
    for col in delinquency_cols:
        df[col] = cap_series(df[col], lower=0, upper=artifacts["delinquency_upper"])

    df["NumberOfDependents"] = cap_series(
        df["NumberOfDependents"], lower=0, upper=artifacts["dependents_upper"]
    )
    df["NumberRealEstateLoansOrLines"] = cap_series(
        df["NumberRealEstateLoansOrLines"], lower=0, upper=artifacts["real_estate_upper"]
    )
    df["NumberOfOpenCreditLinesAndLoans"] = cap_series(
        df["NumberOfOpenCreditLinesAndLoans"], lower=0, upper=artifacts["open_credit_upper"]
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates additional features that may help model performance.
    """
    df = df.copy()

    delinquency_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]

    df["TotalLatePayments"] = df[delinquency_cols].sum(axis=1)
    df["HasAnyLatePayment"] = (df["TotalLatePayments"] > 0).astype("int64")

    df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"])
    df["DebtRatio_log"] = np.log1p(df["DebtRatio"])

    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[18, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "61-100"],
        include_lowest=True,
        ordered=True,
    )

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical engineered features.
    """
    df = df.copy()

    if "AgeGroup" in df.columns:
        df = pd.get_dummies(df, columns=["AgeGroup"], drop_first=True)
        agegroup_cols = [col for col in df.columns if col.startswith("AgeGroup_")]
        if agegroup_cols:
            df[agegroup_cols] = df[agegroup_cols].astype("int64")

    return df


def preprocess_data(
    df: pd.DataFrame,
    create_new_features: bool = True,
    encode_categoricals: bool = True,
    artifacts: Optional[Dict[str, Any]] = None,
    return_artifacts: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full preprocessing pipeline.

    Steps:
    1. Remove unwanted columns
    2. Validate schema
    3. Clean invalid values
    4. Create missing indicators
    5. Fit or use preprocessing artifacts
    6. Apply imputations and outlier handling
    7. Create engineered features
    8. Encode categoricals
    9. Final validations

    If artifacts is None, they will be fitted from the provided dataframe.
    """
    logger.info("Starting preprocessing pipeline...")

    df = remove_unwanted_columns(df)
    validate_columns(df)
    validate_numeric_columns(df)

    # Clean invalid raw values before missing indicators
    df = clean_age(df)
    df = create_missing_indicators(df)

    if artifacts is None:
        artifacts = fit_preprocessing_artifacts(df)
        logger.info("Preprocessing artifacts fitted from current dataframe.")

    df = transform_with_artifacts(df, artifacts)

    if create_new_features:
        df = create_features(df)

    if encode_categoricals:
        df = encode_features(df)

    df = df.sort_index(axis=1)

    validate_no_missing_values(df)

    logger.info("Preprocessing completed successfully. Final shape: %s", df.shape)

    if return_artifacts:
        return df, artifacts
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits dataset into features (X) and target (y).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Saves processed dataframe to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_artifacts(artifacts: Dict[str, Any], output_path: Path) -> None:
    """
    Saves preprocessing artifacts to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)


def load_artifacts(input_path: Path) -> Dict[str, Any]:
    """
    Loads preprocessing artifacts from disk.
    """
    return joblib.load(input_path)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    input_path = BASE_DIR / "data" / "raw" / "cs-training.csv"
    output_data_path = BASE_DIR / "data" / "processed" / "cs-training-processed.csv"
    output_artifacts_path = BASE_DIR / "models" / "preprocessing_artifacts.joblib"

    df_raw = pd.read_csv(input_path)
    df_processed, artifacts = preprocess_data(df_raw, return_artifacts=True)

    save_processed_data(df_processed, output_data_path)
    save_artifacts(artifacts, output_artifacts_path)

    logger.info("Raw shape: %s", df_raw.shape)
    logger.info("Processed shape: %s", df_processed.shape)
    logger.info("Remaining missing values: %s", df_processed.isnull().sum().sum())
    logger.info("Processed data saved to: %s", output_data_path)
    logger.info("Artifacts saved to: %s", output_artifacts_path)

    print("\nProcessed preview:")
    print(df_processed.head())