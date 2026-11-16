# Comprehensive Code Review & Airflow Refactoring Guide

## 1. Executive Summary & Code Assessment

### Overall Rating: **6.5 / 10** (Strong Modeling Foundation, High Refactoring Need)

You have completed solid data science work in your Jupyter notebooks:
- **Comprehensive EDA**: You thoroughly inspected missing values, distributions, invalid logic (`Screen_Time_Hours < Social_Media_Hours`), and target correlations.
- **Systematic Benchmarking**: You evaluated 7 diverse classifiers (`ExtraTrees`, `DecisionTree`, `RandomForest`, `HistGradientBoosting`, `LogisticRegressionCV`, `XGBoost`, `SVC`) with proper classification metrics.
- **Hyperparameter Optimization & Persistence**: You performed grid search on the best-performing model (`LogisticRegression`) and saved the estimator.

However, moving from exploratory notebooks into an automated production pipeline (such as an **Apache Airflow DAG**) requires addressing critical structural issues, severe bugs in `src/`, and data leakage.

---
### Key Issues Summary Table

| Category | Issue Detected | Severity | Impact |
| :--- | :--- | :--- | :--- |
| **Logic Bug** | `src/clean.py`: `fix_screen_time()` filters out valid rows instead of fixing them | 🚨 **Critical** | Deletes 95%+ of dataset during cleaning. |
| **ML Leakage** | Imputation & Encoding performed on entire dataset prior to train/test split | 🚨 **Critical** | Overoptimistic evaluation, target leakage, breaks on new data. |
| **Inference Gap** | Saved model (`joblib`) only contains `StandardScaler` + `LogisticRegression`; encoders were left behind | 🚨 **Critical** | The saved model cannot predict on raw data (expects pre-encoded integers). |
| **Code Anti-pattern** | Functions in `src/clean.py` modify in-place and return string messages (`'unusefull column are deleted!'`) | ⚠️ **High** | Chaining functions (`df = drop_columns(df)`) replaces data with strings. |
| **Airflow Incompatibility** | `src/eda.py` contains `plt.show()` and GUI plotting code | ⚠️ **High** | Airflow DAGs run headless; GUI calls will hang or crash DAG worker tasks. |
| **Broken Imports** | `src/preprocess.py` tries to import `load_raw_data` from `eda` (which doesn't exist) | ⚠️ **High** | Immediate `ImportError` when executed. |
| **Path Hardcoding** | Hardcoded relative paths like `../data/...` throughout | ⚠️ **Medium** | Fails when invoked from repo root or Airflow execution directory. |

---

## 2. Order of Operations: What to Change & Why

### The Current vs. Correct Order of Operations

```
CURRENT ORDER (Notebook / src)                    CORRECT PRODUCTION ORDER (Airflow / Pipeline)
==============================                    =============================================
1. Load Raw CSV                                   1. Load Raw CSV
2. Fix screen time logic                          2. Clean Raw Data (fix screen time, drop Student_ID)
3. Impute missing values (WHOLE DATASET) <---❌  3. Separate Features (X) & Target (y)
4. OrdinalEncode & get_dummies (WHOLE DATASET) <-❌ 4. Train/Test Split (Stratified) <--- SPLIT FIRST!
5. Drop Age & Student_ID                          5. Fit Full Pipeline on X_train ONLY:
6. Train/Test Split                                  ├── Imputation (Iterative / Median / Mode)
7. Scale & Fit Models                                ├── Encoding (OrdinalEncoder + OneHotEncoder)
8. Save best model (missing encoders!)               └── Scaler + Model
                                                  6. Evaluate Pipeline on X_test (Truly unseen data)
                                                  7. Save ENTIRE Pipeline (Preprocessors + Model)
```

### Detailed Explanations of Order Changes

#### 1. Why Imputing and Encoding Before Train/Test Split is Data Leakage
- **Imputation Leakage**: In `cleaning.ipynb` and `src/clean.py`, `IterativeImputer` was fitted on the entire dataset *including the target column `Mental_Health_Risk`*. When the imputer uses target labels and test rows to estimate missing feature values, information from the future test set leaks into the training data.
- **Encoding Leakage**: Using `pd.get_dummies()` and `OrdinalEncoder` on the full dataset leaks the presence and frequencies of categories. If a rare category only appears in the test set or new incoming batch, `get_dummies` will produce different column counts and crash your model.
- **Rule**: Never call `.fit()` or `.fit_transform()` on test data. Always split into `train` and `test` **first**, fit the transformers **only on `X_train`**, and call `.transform()` on `X_test`.

#### 2. Why Encoders Must Be Bundled Inside the Saved Pipeline
In `training_process.ipynb`:
```python
# You did this:
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])
joblib.dump(grid_search.best_estimator_, '../model/mental_health_LR_model.joblib')
```
Here, `best_estimator_` only knows how to scale numbers and predict. The `OrdinalEncoder` for `['Physical_Activity', 'Diet_Quality', ...]` and the dummy encoding for `Gender` are **lost in memory**. When an API or DAG wants to predict a new student's risk, passing raw features will fail.
- **Fix**: Use scikit-learn's `ColumnTransformer` to package imputation, categorical encoding, ordinal encoding, and scaling together with the model into **one single pipeline artifact**.

---

## 3. Function Design: What Should & Shouldn't Be a Function

Not everything should be a function. Writing 1-line functions that perform in-place mutation and return status strings creates maintenance nightmares.

### Rules for Writing Functions in ML Engineering
1. **Pure Data Transformations**: A function taking a DataFrame should return a transformed DataFrame (`df -> df`). It should **not** modify global variables or return status strings.
2. **Deterministic & Parameterized**: File paths, column names, and random states should be passed as parameters or imported from a config module, never hardcoded.
3. **Single Responsibility**: One function should do one clear job (e.g., clean raw data, build preprocessor, train model, compute metrics).

### What SHOULD Be a Function

| Function Name | Location | Inputs | Outputs | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `load_data` | `src/clean.py` | `filepath: Path or str` | `pd.DataFrame` | Safely reads raw CSV into a DataFrame. |
| `clean_raw_data` | `src/clean.py` | `df: pd.DataFrame` | `pd.DataFrame` | Fixes screen time, drops `Student_ID`, checks types. |
| `save_data` | `src/clean.py` | `df: pd.DataFrame, path: str` | `None` | Saves cleaned dataset to CSV/Parquet. |
| `build_preprocessor` | `src/pipeline.py` | None (or feature dict) | `ColumnTransformer` | Defines imputer, encoders, and scalers per column type. |
| `build_full_pipeline`| `src/pipeline.py` | `model, preprocessor` | `Pipeline` | Glues preprocessor and model into a single estimator. |
| `split_data` | `src/train.py` | `df: pd.DataFrame` | `X_train, X_test, y_train, y_test` | Separates target and creates stratified splits. |
| `tune_and_train` | `src/train.py` | `X_train, y_train` | `best_pipeline: Pipeline` | Runs GridSearchCV and fits final model. |
| `evaluate_model` | `src/evaluate.py` | `model, X_test, y_test` | `dict` (metrics) | Computes accuracy, F1-score, confusion matrix. |
| `save_model_artifacts`| `src/train.py` | `pipeline, path: str` | `None` | Serializes fitted pipeline using `joblib`. |

### What SHOULD NOT Be a Function
- **Status message wrappers**: Functions like `def drop_columns(df): ... return 'unusefull column are deleted!'` should not exist. Instead, let `clean_raw_data(df)` drop non-predictive columns and return the cleaned DataFrame.
- **Interactive EDA and Plotting (`plt.show()`)**: EDA is exploratory. In an automated Airflow DAG, interactive plots block the process and fail because server workers lack display environments.
- **Top-level execution script logic**: Running loops over 7 models during rapid experimentation does not need to be 7 separate functions; it belongs in a model selection experiment script or notebook.

---

## 4. File Division: Is Your Current Structure Correct?

### Critique of Current Division

```
src/
├── clean.py       <-- Mixed: has data loading, string-returning droppers, bug in screen time, and imputer
├── eda.py         <-- Has plt.show(); unusable in Airflow DAG
├── preprocess.py  <-- Has broken import (3 lines of non-working code)
└── train.py       <-- Empty (0 bytes)
```

1. **`eda.py` should NOT be in the DAG path**: Keep EDA in the `notebook/` folder for human analysis, or convert it to a headless reporting script (saving PNGs/HTML reports to an output folder), but keep it out of the core pipeline.
2. **Confusion between `clean.py` and `preprocess.py`**:
   - **Cleaning**: Fix data collection anomalies (e.g., `Screen_Time < Social_Media`, dropping ID columns). This is row/column level hygiene.
   - **Preprocessing**: Transforming features for machine learning (imputing NaNs, one-hot encoding, scaling). This should live inside `pipeline.py` using scikit-learn transformers.
3. **Missing Modules**:
   - `config.py`: You have column lists and file paths scattered across notebooks and python files. Centralize them.
   - `evaluate.py`: Model validation logic should be distinct from training logic so Airflow can run a validation gate before model deployment.

### Recommended Production File Structure

```text
MentalHealth_SC_1/
├── data/
│   ├── student_mental_health.csv      # Raw source data (read-only)
│   └── cleaned_ds.csv                 # Cleaned data (output of clean task)
├── model/
│   ├── mental_health_LR_model.joblib  # Full pipeline artifact
│   └── metrics.json                   # Test metrics
├── notebook/
│   ├── cleaning.ipynb                 # EDA & experimentation (keep for notes)
│   └── training_process.ipynb         # Model benchmark research
├── src/
│   ├── __init__.py
│   ├── config.py                      # Constants, column definitions, paths
│   ├── clean.py                       # Pure data cleaning functions
│   ├── pipeline.py                    # Preprocessing ColumnTransformer & Pipeline
│   ├── train.py                       # Training, tuning & saving pipeline
│   └── evaluate.py                    # Evaluation metrics & reports
├── dags/
│   └── mental_health_dag.py           # Airflow DAG definition
└── README.md
```

---

## 5. File-by-File Action Plan: What to Do, Change, and Add

### 1. `src/config.py` (New File to Add)
Create this file to centralize all project constants. This avoids hardcoding column names and paths across multiple scripts.

```python
"""
Configuration file: Centralizes paths, random seeds, and feature definitions.
"""
from pathlib import Path

# Paths (Anchor to Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

RAW_DATA_PATH = DATA_DIR / "student_mental_health.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_ds.csv"
MODEL_PATH = MODEL_DIR / "mental_health_LR_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Feature Categorization
TARGET_COL = "Mental_Health_Risk"
DROP_COLS = ["Student_ID"]  # Note: Decide whether to keep or drop Age

# Features for Preprocessing
NUMERICAL_COLS = [
    "Age",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
    "Year_of_Study"
]

ONE_HOT_COLS = ["Gender"]

ORDINAL_COLS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
    "Part_Time_Job": ["No", "Yes"],
    "Previous_Mental_Health_History": ["No", "Yes"]
}

# Modeling
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

---

### 2. `src/clean.py` (What to Fix & What to Keep)

#### Issues in your current `clean.py`:
- **Line 10**: `load_raw_data()` loads `cleaned_ds.csv` instead of `student_mental_health.csv`.
- **Line 17**: `drop_columns()` returns string `'unusefull column are deleted!'`.
- **Line 26**: `fix_screen_time()` has a fatal bug:
  ```python
  # YOUR CURRENT CODE (WRONG - drops all normal records):
  def fix_screen_time(df):
      df = df[(df['Screen_Time_Hours'] < df['Social_Media_Hours'])]
      return df
  ```
- **Line 35**: `impute_missing_values()` does full dataset imputation here. **Move this to `pipeline.py`** to prevent data leakage.

#### How `src/clean.py` should look:
```python
"""
Data cleaning module: Handles raw ingestion and logical data hygiene.
"""
import pandas as pd
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH, DROP_COLS

def load_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw dataset from disk."""
    return pd.read_csv(path)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data:
    1. Fixes illogical screen time (Screen Time cannot be lower than Social Media Time).
    2. Drops non-predictive identifier columns.
    """
    df = df.copy()

    # 1. Correct screen time where social media hours exceed total screen hours
    condition = df["Screen_Time_Hours"] < df["Social_Media_Hours"]
    df.loc[condition, "Screen_Time_Hours"] = df.loc[condition, "Social_Media_Hours"]

    # 2. Drop identifiers (e.g. Student_ID)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df

def save_data(df: pd.DataFrame, path=CLEANED_DATA_PATH) -> None:
    """Saves cleaned DataFrame."""
    df.to_csv(path, index=False)
```

---

### 3. `src/eda.py` (What to Do With It)
- **Do NOT import or call `eda.py` from your Airflow DAG**.
- Keep `eda.py` or the notebook for offline exploration.
- If you need automated data quality checks in Airflow, do not use plots. Instead, write validation checks:
  - Check that missing percentage is below 10%.
  - Check that expected columns exist.
  - Check that numerical values fall within realistic ranges (e.g., $GPA \le 4.0$).

---

### 4. `src/pipeline.py` (Replaces `preprocess.py`)
Replace the non-working `src/preprocess.py` with `src/pipeline.py`.
This module builds the scikit-learn `ColumnTransformer` and the end-to-end `Pipeline`.

```python
"""
Pipeline module: Constructs full scikit-learn pipeline (Imputation + Encoders + Scaler + Model).
Prevents data leakage and ensures seamless inference.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

from src.config import NUMERICAL_COLS, ONE_HOT_COLS, ORDINAL_COLS, RANDOM_STATE

def build_preprocessor() -> ColumnTransformer:
    """Creates a ColumnTransformer to handle numerical and categorical columns."""
    
    # 1. Numerical Pipeline: Impute with median, then scale
    num_cols = [c for c in NUMERICAL_COLS if c != "Age"] # Adjust if Age is dropped
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. One-Hot Pipeline: Impute with most frequent, then one-hot encode
    one_hot_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # 3. Ordinal Pipeline: Impute with most frequent, then ordinal encode
    ordinal_categories = [ORDINAL_COLS[col] for col in ORDINAL_COLS]
    ordinal_cols = list(ORDINAL_COLS.keys())
    
    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', one_hot_transformer, ONE_HOT_COLS),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )
    return preprocessor

def create_full_pipeline(model=None) -> Pipeline:
    """Combines preprocessor and estimator into one deployable object."""
    if model is None:
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

    preprocessor = build_preprocessor()
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline
```

---

### 5. `src/train.py` (Full Implementation)
Currently, `src/train.py` is empty. Implement functions to split data, run grid search, and save the pipeline.

```python
"""
Training module: Handles data splitting, hyperparameter optimization, and saving artifacts.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from src.config import (
    CLEANED_DATA_PATH,
    MODEL_PATH,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE
)
from src.pipeline import create_full_pipeline

def split_dataset(df: pd.DataFrame):
    """Splits cleaned DataFrame into train and test sets."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_and_optimize(X_train, y_train):
    """Tunes hyperparameters for the full pipeline using GridSearchCV."""
    pipeline = create_full_pipeline()

    param_grid = {
        'model__C': [0.01, 0.1, 1.0, 10.0],
        'model__solver': ['lbfgs', 'saga']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def save_model(model_pipeline, path=MODEL_PATH):
    """Saves the complete pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, path)
```

---

### 6. `src/evaluate.py` (New File to Add)
Provides functions to evaluate the trained model on test data and save metrics.

```python
"""
Evaluation module: Computes metrics on test set and saves validation reports.
"""
import json
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.config import MODEL_PATH, METRICS_PATH

def evaluate_model(model_path=MODEL_PATH, X_test=None, y_test=None, metrics_path=METRICS_PATH):
    """Loads pipeline, predicts on test data, and logs metrics."""
    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": acc,
        "confusion_matrix": matrix,
        "classification_report": report
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
```

---

## 6. How Your Airflow DAG Will Look

When you organize your functions this way, your Airflow DAG becomes lightweight, clean, and readable. It only orchestrates tasks and does not contain dirty data science logic.

```python
"""
Example Airflow DAG: dags/mental_health_dag.py
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.clean import load_data, clean_raw_data, save_data
from src.train import split_dataset, train_and_optimize, save_model
from src.evaluate import evaluate_model
from src.config import CLEANED_DATA_PATH

def task_clean_data():
    df = load_data()
    df_clean = clean_raw_data(df)
    save_data(df_clean)

def task_train_model():
    import pandas as pd
    df = pd.read_csv(CLEANED_DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    # Save test set temporarily or in artifacts for evaluation task
    X_test.to_parquet("data/X_test.parquet")
    y_test.to_frame().to_parquet("data/y_test.parquet")

    best_pipeline = train_and_optimize(X_train, y_train)
    save_model(best_pipeline)

def task_evaluate_model():
    import pandas as pd
    X_test = pd.read_parquet("data/X_test.parquet")
    y_test = pd.read_parquet("data/y_test.parquet").squeeze()

    metrics = evaluate_model(X_test=X_test, y_test=y_test)
    print(f"Model Accuracy: {metrics['accuracy']:.4f}")
    
    # Quality gate: fail task if model accuracy drops below acceptable threshold
    if metrics['accuracy'] < 0.70:
        raise ValueError(f"Model accuracy {metrics['accuracy']} is below quality threshold!")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'student_mental_health_training_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
    catchup=False
) as dag:

    clean_step = PythonOperator(
        task_id='clean_data',
        python_callable=task_clean_data
    )

    train_step = PythonOperator(
        task_id='train_and_tune_model',
        python_callable=task_train_model
    )

    evaluate_step = PythonOperator(
        task_id='evaluate_model',
        python_callable=task_evaluate_model
    )

    clean_step >> train_step >> evaluate_step
```

---

## 7. Step-by-Step Transition Roadmap

To successfully refactor your project without breaking anything, follow these steps:

1. **Step 1: Fix the `fix_screen_time` bug in `src/clean.py`**
   - Replace the filtering line with `.loc[condition, 'Screen_Time_Hours'] = df.loc[condition, 'Social_Media_Hours']`.
   - Ensure all cleaning functions return the modified `DataFrame`.

2. **Step 2: Add `src/config.py`**
   - Centralize paths using `pathlib.Path` and define feature column lists.

3. **Step 3: Build `src/pipeline.py`**
   - Implement `ColumnTransformer` bundling median/mode imputation, `OrdinalEncoder`, `OneHotEncoder`, and `StandardScaler`.
   - Pair with `LogisticRegression` inside a `Pipeline`.

4. **Step 4: Implement `src/train.py`**
   - Add `split_dataset()` and `train_and_optimize()`.
   - Verify that `joblib.dump()` saves the full pipeline (preprocessor + model).

5. **Step 5: Implement `src/evaluate.py`**
   - Add evaluation logic that reads test data and saves `metrics.json`.

6. **Step 6: Test Locally from Terminal**
   - Run the sequence from Python scripts or a test runner to ensure no `ImportError` or path issues arise:
     ```bash
     python -c "from src.clean import load_data, clean_raw_data, save_data; df = clean_raw_data(load_data()); save_data(df)"
     ```

7. **Step 7: Build the Airflow DAG**
   - Create your DAG file pointing to the tested `src` functions.

