# ML Experiment Autopilot

**ML Experiment Autopilot** is an autonomous agent that designs, executes, and iterates on machine learning experiments without human supervision. It uses Google's Gemini 3 API with Thought Signatures to maintain reasoning continuity across hundreds of API calls.

---

## Quick Demo

**Prerequisites:** Python 3.9+, a [Gemini API key](https://aistudio.google.com/apikey)

### 1. Set up the environment

```bash
cd ml-experiment-autopilot

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and replace `your_gemini_api_key_here` with your actual Gemini API key:

```
GEMINI_API_KEY=your_actual_key_here
```

### 3. Verify the CLI works

```bash
python -m src.main --help
```

You should see the available commands and flags listed.

### 4. Run the autopilot

**Regression (House Prices) — primary demo:**

```bash
python -m src.main run \
  --data data/sample/house_prices_train.csv \
  --target SalePrice \
  --task regression \
  --max-iterations 3 \
  --verbose
```

**With constraints file** (guides Gemini to prefer tree-based models, log-transform target, etc.):

```bash
python -m src.main run \
  --data data/sample/house_prices_train.csv \
  --target SalePrice \
  --task regression \
  --constraints data/sample/constraints.md \
  --max-iterations 3 \
  --verbose
```

**Classification (Titanic):**

```bash
python -m src.main run \
  --data data/sample/titanic_train.csv \
  --target Survived \
  --task classification \
  --max-iterations 3 \
  --verbose
```

**Quick single-line command** (short flags):

```bash
python -m src.main run -d data/sample/house_prices_train.csv -t SalePrice --task regression -n 3 -v
```

---

## What to expect

The autopilot runs through these phases, printing Rich-formatted output to your terminal:

1. **Data Profiling** — Analyzes dataset shape, column types, missing values, statistics
2. **Baseline Model** — Trains a simple model (LinearRegression / LogisticRegression) to establish baseline performance
3. **Iteration Loop** (repeats up to `--max-iterations` times):
   - **Experiment Design** — Gemini designs the next experiment (model, hyperparameters, preprocessing)
   - **Code Generation** — Generates a training script from templates
   - **Experiment Execution** — Runs the script in a subprocess, captures metrics
   - **Results Analysis** — Compares metrics against baseline/best, detects trends
   - **Hypothesis Generation** — Generates ranked hypotheses for the next iteration
4. **Summary** — Prints final results: best model, best metric, total iterations

In verbose mode (`-v`), you will also see Gemini's reasoning, thought signature context, key observations, and hypothesis details.

---

## View results in MLflow

After the run completes, you can explore all experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri file:./outputs/mlruns
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Run tests

```bash
pytest tests/ -v
```
