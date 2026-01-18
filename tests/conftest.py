"""Pytest fixtures for ML Experiment Autopilot tests."""

import os
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_regression_data():
    """Create a sample regression dataset."""
    np.random.seed(42)
    n = 100

    data = {
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.choice(['A', 'B', 'C'], n),
        'feature4': np.random.randint(1, 10, n),
    }

    # Create target with some correlation
    data['target'] = (
        data['feature1'] * 2 +
        data['feature2'] * 0.5 +
        data['feature4'] * 0.3 +
        np.random.randn(n) * 0.1
    )

    return pd.DataFrame(data)


@pytest.fixture
def sample_classification_data():
    """Create a sample classification dataset."""
    np.random.seed(42)
    n = 100

    data = {
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.choice(['X', 'Y', 'Z'], n),
        'feature4': np.random.randint(1, 10, n),
    }

    # Create binary target
    prob = 1 / (1 + np.exp(-(data['feature1'] + data['feature2'])))
    data['target'] = (prob > 0.5).astype(int)

    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing():
    """Create a sample dataset with missing values."""
    np.random.seed(42)
    n = 100

    data = {
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.choice(['A', 'B', 'C', None], n),
        'target': np.random.randn(n),
    }

    # Add some missing values
    df = pd.DataFrame(data)
    df.loc[df.sample(frac=0.1).index, 'feature1'] = np.nan
    df.loc[df.sample(frac=0.15).index, 'feature2'] = np.nan

    return df


@pytest.fixture
def temp_data_file(sample_regression_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_regression_data.to_csv(f.name, index=False)
        yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def temp_classification_file(sample_classification_data):
    """Create a temporary CSV file with classification data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_classification_data.to_csv(f.name, index=False)
        yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response for experiment design."""
    return {
        "experiment_name": "test_random_forest",
        "hypothesis": "Testing if RandomForest improves over baseline",
        "model_type": "RandomForestRegressor",
        "model_params": {
            "n_estimators": 100,
            "max_depth": 10,
        },
        "preprocessing": {
            "missing_values": "median",
            "scaling": "standard",
            "encoding": "onehot",
        },
        "reasoning": "Based on the data profile, tree-based models should perform well",
    }


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, temp_output_dir):
    """Set up test environment variables."""
    # Use a fake API key for tests that don't actually call the API
    monkeypatch.setenv("GEMINI_API_KEY", "test_api_key_for_testing")
