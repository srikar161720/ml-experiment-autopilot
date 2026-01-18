#!/bin/bash
# Download sample datasets for ML Experiment Autopilot

set -e

echo "Downloading sample datasets..."

# Create data directory
mkdir -p data/sample

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check for Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Warning: Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo "Please set up Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Move kaggle.json to ~/.kaggle/"
    echo ""
    echo "Creating sample data manually instead..."

    # Create a simple sample dataset for testing
    python3 << 'EOF'
import pandas as pd
import numpy as np

# Create a simple house prices sample
np.random.seed(42)
n = 100

data = {
    'OverallQual': np.random.randint(1, 11, n),
    'GrLivArea': np.random.randint(500, 3000, n),
    'GarageCars': np.random.randint(0, 4, n),
    'TotalBsmtSF': np.random.randint(0, 2000, n),
    'FullBath': np.random.randint(1, 4, n),
    'YearBuilt': np.random.randint(1950, 2020, n),
    'Neighborhood': np.random.choice(['A', 'B', 'C', 'D'], n),
}

# Create correlated target
data['SalePrice'] = (
    data['OverallQual'] * 20000 +
    data['GrLivArea'] * 50 +
    data['GarageCars'] * 10000 +
    np.random.normal(0, 20000, n)
).astype(int)

df = pd.DataFrame(data)
df.to_csv('data/sample/house_prices_train.csv', index=False)
print(f"Created house_prices_train.csv with {len(df)} rows")

# Create Titanic sample
data_titanic = {
    'Pclass': np.random.randint(1, 4, n),
    'Sex': np.random.choice(['male', 'female'], n),
    'Age': np.random.randint(1, 80, n),
    'SibSp': np.random.randint(0, 5, n),
    'Parch': np.random.randint(0, 3, n),
    'Fare': np.random.uniform(5, 500, n),
    'Embarked': np.random.choice(['S', 'C', 'Q'], n),
}

# Create correlated survival
survival_prob = (
    (data_titanic['Pclass'] == 1).astype(float) * 0.3 +
    (np.array(data_titanic['Sex']) == 'female').astype(float) * 0.4 +
    (np.array(data_titanic['Age']) < 18).astype(float) * 0.2 +
    np.random.uniform(0, 0.3, n)
)
data_titanic['Survived'] = (survival_prob > 0.5).astype(int)

df_titanic = pd.DataFrame(data_titanic)
df_titanic.to_csv('data/sample/titanic_train.csv', index=False)
print(f"Created titanic_train.csv with {len(df_titanic)} rows")
EOF

    echo ""
    echo "Sample datasets created!"
    exit 0
fi

# Download House Prices dataset
echo "Downloading House Prices dataset..."
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/sample/
unzip -o data/sample/house-prices-advanced-regression-techniques.zip -d data/sample/
rm -f data/sample/house-prices-advanced-regression-techniques.zip
mv data/sample/train.csv data/sample/house_prices_train.csv 2>/dev/null || true

# Download Titanic dataset
echo "Downloading Titanic dataset..."
kaggle competitions download -c titanic -p data/sample/
unzip -o data/sample/titanic.zip -d data/sample/
rm -f data/sample/titanic.zip
mv data/sample/train.csv data/sample/titanic_train.csv 2>/dev/null || true

echo ""
echo "Datasets downloaded to data/sample/"
ls -la data/sample/
