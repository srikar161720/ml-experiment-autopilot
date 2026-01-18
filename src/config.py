"""Configuration management for ML Experiment Autopilot."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
TEMPLATES_DIR = PROJECT_ROOT / "templates"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"
MLRUNS_DIR = OUTPUTS_DIR / "mlruns"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"


@dataclass
class GeminiConfig:
    """Configuration for Gemini API."""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = "gemini-3-pro-preview"  # Gemini 3 Pro Preview model
    temperature: float = 1.0  # REQUIRED - lower values degrade reasoning
    default_thinking_level: str = "high"
    max_retries: int = 3
    retry_delay: float = 1.0  # Initial delay in seconds for exponential backoff

    def __post_init__(self):
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )


@dataclass
class ExperimentDefaults:
    """Default values for experiment configuration."""

    max_iterations: int = 20
    time_budget: int = 3600  # seconds
    plateau_threshold: int = 3  # consecutive iterations without improvement
    improvement_threshold: float = 0.005  # 0.5% relative improvement
    experiment_timeout: int = 300  # seconds per experiment


@dataclass
class Config:
    """Main configuration for the autopilot."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    defaults: ExperimentDefaults = field(default_factory=ExperimentDefaults)
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", str(MLRUNS_DIR))
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    verbose: bool = False


def get_config(verbose: bool = False) -> Config:
    """Get the application configuration.

    Args:
        verbose: Whether to enable verbose output.

    Returns:
        Config object with all settings.
    """
    config = Config()
    config.verbose = verbose
    return config


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        TEMPLATES_DIR,
        OUTPUTS_DIR,
        EXPERIMENTS_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        MLRUNS_DIR,
        DATA_DIR,
        SAMPLE_DATA_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on module import
ensure_directories()
