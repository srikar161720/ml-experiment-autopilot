"""Tests for the DataProfiler."""

import pytest
import pandas as pd
import numpy as np

from src.execution.data_profiler import DataProfiler
from src.orchestration.state import DataProfile


class TestDataProfiler:
    """Test cases for DataProfiler."""

    def test_load_csv(self, temp_data_file):
        """Test loading a CSV file."""
        profiler = DataProfiler(temp_data_file, "target", "regression")
        df = profiler.load_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "target" in df.columns

    def test_profile_basic(self, temp_data_file):
        """Test basic profiling."""
        profiler = DataProfiler(temp_data_file, "target", "regression")
        profile = profiler.profile()

        assert isinstance(profile, DataProfile)
        assert profile.n_rows == 100
        assert profile.n_columns == 5
        assert profile.target_column == "target"
        assert profile.target_type == "numeric"

    def test_profile_columns(self, temp_data_file):
        """Test column detection."""
        profiler = DataProfiler(temp_data_file, "target", "regression")
        profile = profiler.profile()

        assert "feature1" in profile.columns
        assert "feature2" in profile.columns
        assert "feature3" in profile.columns
        assert "feature4" in profile.columns
        assert "target" in profile.columns

    def test_numeric_columns(self, temp_data_file):
        """Test numeric column detection."""
        profiler = DataProfiler(temp_data_file, "target", "regression")
        profile = profiler.profile()

        assert "feature1" in profile.numeric_columns
        assert "feature2" in profile.numeric_columns
        assert "feature4" in profile.numeric_columns

    def test_categorical_columns(self, temp_data_file):
        """Test categorical column detection."""
        profiler = DataProfiler(temp_data_file, "target", "regression")
        profile = profiler.profile()

        assert "feature3" in profile.categorical_columns

    def test_numeric_stats(self, temp_data_file):
        """Test numeric statistics calculation."""
        profiler = DataProfiler(temp_data_file, "target", "regression")
        profile = profiler.profile()

        assert "feature1" in profile.numeric_stats
        stats = profile.numeric_stats["feature1"]

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "50%" in stats  # median

    def test_missing_values(self, temp_data_file, sample_data_with_missing):
        """Test missing value detection."""
        import tempfile
        import os

        # Create temp file with missing data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data_with_missing.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            profiler = DataProfiler(temp_path, "target", "regression")
            profile = profiler.profile()

            # Should detect missing values
            assert sum(profile.missing_values.values()) > 0
        finally:
            os.unlink(temp_path)

    def test_target_not_found(self, temp_data_file):
        """Test error when target column is missing."""
        profiler = DataProfiler(temp_data_file, "nonexistent", "regression")

        with pytest.raises(ValueError, match="not found"):
            profiler.profile()

    def test_classification_task(self, temp_classification_file):
        """Test profiling for classification task."""
        profiler = DataProfiler(temp_classification_file, "target", "classification")
        profile = profiler.profile()

        assert profile.target_type == "categorical"
        assert "n_classes" in profile.target_stats
        assert "class_distribution" in profile.target_stats

    def test_profile_summary(self, temp_data_file):
        """Test human-readable summary."""
        profiler = DataProfiler(temp_data_file, "target", "regression")
        summary = profiler.get_profile_summary()

        assert isinstance(summary, str)
        assert "target" in summary.lower()
        assert "regression" in summary.lower()

    def test_to_dict(self, temp_data_file):
        """Test conversion to dictionary."""
        profiler = DataProfiler(temp_data_file, "target", "regression")
        profile_dict = profiler.to_dict()

        assert isinstance(profile_dict, dict)
        assert "n_rows" in profile_dict
        assert "columns" in profile_dict
