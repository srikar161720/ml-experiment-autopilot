# ML Experiment Autopilot

**ML Experiment Autopilot** is an autonomous agent that designs, executes, and iterates on machine learning experiments without human supervision. It uses Google's Gemini 3 API with Thought Signatures to maintain reasoning continuity across hundreds of API calls.

### Commands to test the autopilot
From your terminal in the root directory of the project, run the following commands:

```bash
# 1. (Optional) Create and activate a virtual environment if you haven't already
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (skip if already done)
pip install -r requirements.txt

# 3. Verify the CLI is working
python -m src.main --help

# 4. Run the autopilot on sample data (3 iterations, verbose mode)
python -m src.main run \
  --data data/sample/house_prices_train.csv \
  --target SalePrice \
  --task regression \
  --max-iterations 3 \
  --verbose
```

Quick single command if dependencies are already installed:
```bash
python -m src.main run -d data/sample/house_prices_train.csv -t SalePrice --task regression -n 3 -v
```

After the run, you can view MLflow results:
```bash
mlflow ui --backend-store-uri file:./outputs/mlruns
```
Then go to `http://127.0.0.1:5000` in your browser to view the results.
