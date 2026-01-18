"""Data profiling for ML datasets."""

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.orchestration.state import DataProfile, TaskType


class DataProfiler:
    """Profile datasets for ML experimentation.

    Analyzes the dataset to provide:
    - Schema information (columns, types)
    - Statistical summaries
    - Missing value analysis
    - Categorical variable analysis
    - Target variable analysis
    """

    def __init__(self, data_path: Path, target_column: str, task_type: str):
        """Initialize the data profiler.

        Args:
            data_path: Path to the dataset file (CSV or Parquet).
            target_column: Name of the target column.
            task_type: 'classification' or 'regression'.
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.task_type = TaskType(task_type)
        self.df: pd.DataFrame = None

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from file.

        Returns:
            Loaded pandas DataFrame.

        Raises:
            ValueError: If file format is not supported.
        """
        suffix = self.data_path.suffix.lower()

        if suffix == ".csv":
            self.df = pd.read_csv(self.data_path)
        elif suffix == ".parquet":
            self.df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use CSV or Parquet.")

        return self.df

    def profile(self) -> DataProfile:
        """Generate a complete profile of the dataset.

        Returns:
            DataProfile with all analysis results.

        Raises:
            ValueError: If target column is not found in dataset.
        """
        if self.df is None:
            self.load_data()

        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Get column types
        column_types = self._get_column_types()
        numeric_columns = self._get_numeric_columns()
        categorical_columns = self._get_categorical_columns()

        # Determine target type
        target_type = self._get_target_type()

        # Calculate missing values
        missing_values = self._get_missing_values()
        missing_percentages = self._get_missing_percentages()

        # Get statistics
        numeric_stats = self._get_numeric_stats(numeric_columns)
        categorical_stats = self._get_categorical_stats(categorical_columns)
        target_stats = self._get_target_stats()

        return DataProfile(
            n_rows=len(self.df),
            n_columns=len(self.df.columns),
            columns=list(self.df.columns),
            column_types=column_types,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            target_column=self.target_column,
            target_type=target_type,
            missing_values=missing_values,
            missing_percentages=missing_percentages,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            target_stats=target_stats,
        )

    def _get_column_types(self) -> dict[str, str]:
        """Get the dtype of each column as a string."""
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def _get_numeric_columns(self) -> list[str]:
        """Get list of numeric column names."""
        return list(self.df.select_dtypes(include=[np.number]).columns)

    def _get_categorical_columns(self) -> list[str]:
        """Get list of categorical column names."""
        # Include object, category, and bool types
        categorical = list(
            self.df.select_dtypes(include=["object", "category", "bool"]).columns
        )
        return categorical

    def _get_target_type(self) -> str:
        """Determine the type of the target column."""
        target_series = self.df[self.target_column]

        if self.task_type == TaskType.CLASSIFICATION:
            return "categorical"
        elif np.issubdtype(target_series.dtype, np.number):
            return "numeric"
        else:
            return "categorical"

    def _get_missing_values(self) -> dict[str, int]:
        """Get count of missing values per column."""
        return {col: int(self.df[col].isna().sum()) for col in self.df.columns}

    def _get_missing_percentages(self) -> dict[str, float]:
        """Get percentage of missing values per column."""
        n_rows = len(self.df)
        return {
            col: round(self.df[col].isna().sum() / n_rows * 100, 2)
            for col in self.df.columns
        }

    def _get_numeric_stats(self, numeric_columns: list[str]) -> dict[str, dict[str, float]]:
        """Get statistical summary for numeric columns."""
        stats = {}
        for col in numeric_columns:
            if col == self.target_column:
                continue  # Target stats handled separately

            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "25%": float(series.quantile(0.25)),
                "50%": float(series.quantile(0.50)),
                "75%": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skew": float(series.skew()) if len(series) > 2 else 0.0,
            }
        return stats

    def _get_categorical_stats(
        self, categorical_columns: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Get statistics for categorical columns."""
        stats = {}
        for col in categorical_columns:
            if col == self.target_column:
                continue  # Target stats handled separately

            series = self.df[col].dropna()
            value_counts = series.value_counts()

            stats[col] = {
                "n_unique": int(series.nunique()),
                "top_values": value_counts.head(10).to_dict(),
                "cardinality_ratio": round(series.nunique() / len(series), 4)
                if len(series) > 0
                else 0,
            }
        return stats

    def _get_target_stats(self) -> dict[str, Any]:
        """Get statistics for the target column."""
        target_series = self.df[self.target_column].dropna()

        if self.task_type == TaskType.CLASSIFICATION:
            value_counts = target_series.value_counts()
            return {
                "n_classes": int(target_series.nunique()),
                "class_distribution": value_counts.to_dict(),
                "class_balance": round(
                    value_counts.min() / value_counts.max(), 4
                )
                if value_counts.max() > 0
                else 1.0,
            }
        else:
            # Regression
            return {
                "mean": float(target_series.mean()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "25%": float(target_series.quantile(0.25)),
                "50%": float(target_series.quantile(0.50)),
                "75%": float(target_series.quantile(0.75)),
                "max": float(target_series.max()),
                "skew": float(target_series.skew()) if len(target_series) > 2 else 0.0,
            }

    def get_profile_summary(self) -> str:
        """Get a human-readable summary of the profile."""
        profile = self.profile()

        summary = [
            f"Dataset: {self.data_path.name}",
            f"Shape: {profile.n_rows} rows x {profile.n_columns} columns",
            f"Target: {profile.target_column} ({profile.target_type})",
            f"Task: {self.task_type.value}",
            "",
            f"Numeric features: {len(profile.numeric_columns)}",
            f"Categorical features: {len(profile.categorical_columns)}",
            "",
        ]

        # Missing values summary
        missing = {k: v for k, v in profile.missing_values.items() if v > 0}
        if missing:
            summary.append(f"Columns with missing values: {len(missing)}")
            for col, count in list(missing.items())[:5]:
                pct = profile.missing_percentages[col]
                summary.append(f"  - {col}: {count} ({pct}%)")
            if len(missing) > 5:
                summary.append(f"  - ... and {len(missing) - 5} more")
        else:
            summary.append("No missing values")

        # Target summary
        summary.append("")
        summary.append("Target variable:")
        for key, value in profile.target_stats.items():
            if isinstance(value, dict) and len(value) > 5:
                summary.append(f"  - {key}: {len(value)} unique values")
            else:
                summary.append(f"  - {key}: {value}")

        return "\n".join(summary)

    def to_dict(self) -> dict:
        """Convert profile to dictionary for JSON serialization."""
        return self.profile().model_dump()
