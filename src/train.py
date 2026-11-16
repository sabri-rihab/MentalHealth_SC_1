import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from src.config import CLEANED_DATA_PATH, MODEL_PATH, TARGET_COL
from src.pipline import create_full_pipline

def split_dataset(df: pd.DataFrame):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        stratify=y,
        test_size=0.2
    )

    return X_train, X_test, y_train, y_test


def train_and_optimize(X_train, y_train):
    pipline = create_full_pipline()
    grid_params = {
        'model__C': [0.01, 0.1, 1.0, 10.0],
        'model__solver': ['saga', 'liblinear', 'lbfgs', 'newton-cg'],
        'model__penalty': ['l1', 'l2', 'elasticnet']
        # 'model__l1_ratio': [0, 0.5, 1],
    }

    grid_search = GridSearchCV(
        param_grid=grid_params,
        estimator=pipline,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def save_model(modle_pipline, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modle_pipline, path)