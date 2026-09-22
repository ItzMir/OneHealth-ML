# src/utils/verification.py
import pandas as pd


def dataset_summary(df: pd.DataFrame, name: str = "dataset") -> None:
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(df.info())


def verify_variables(df: pd.DataFrame, variables: list, strict: bool = True) -> list:
    """
    Confirm that all expected variables exist in the dataframe.
    If strict=False, logs missing variables but does not raise.
    """
    missing = [v for v in variables if v not in df.columns]
    if missing:
        if strict:
            raise ValueError(f"Missing variables: {missing}")
        else:
            print(f"⚠ Missing variables ({len(missing)}): {missing}")
            return missing
    print(f"✅ All {len(variables)} variables verified.")
    return []


def missing_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)
    if missing.empty:
        print("No missing values detected.")
        return
    print("Top columns with missing values:")
    for col, n in missing.items():
        pct = 100 * n / len(df)
        print(f"  {col}: {n} ({pct:.1f}%)")


def categorical_summary(df: pd.DataFrame, columns: list) -> None:
    for col in columns:
        if col not in df.columns:
            print(f"{col}: NOT FOUND")
            continue
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(10))