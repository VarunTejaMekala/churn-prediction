import yaml
import os
import sys
import numpy as np
import pickle

from sklearn.metrics import f1_score             # FIX: was r2_score — wrong metric for classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # FIX: added StratifiedKFold

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as yaml_file:   # FIX: was "rb" — yaml.safe_load needs text mode
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """Save numpy array to .npy file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=False)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """Pickle an object to disk."""
    try:
        logging.info(f"Saving object to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_object(file_path: str) -> object:
    """Load a pickled object from disk."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
    param: dict,
    threshold: float = 0.35,
) -> dict:
    """
    Runs GridSearchCV for each model and returns test F1 scores.

    FIX: replaced r2_score with f1_score — r2 is a regression metric and is
         meaningless (often negative) for binary classification.
    FIX: uses StratifiedKFold to preserve class balance across folds.
    FIX: evaluates with probability threshold (not default 0.5) for churn.
    """
    try:
        report = {}

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for model_name, model in models.items():
            logging.info(f"Running GridSearchCV for: {model_name}")
            para = param[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=cv,
                scoring="f1",       # FIX: score on f1 during CV, not accuracy
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            logging.info(f"{model_name} best params: {gs.best_params_}")

            # Use probability threshold for churn-sensitive evaluation
            if hasattr(best_model, "predict_proba"):
                y_test_prob = best_model.predict_proba(X_test)[:, 1]
                y_test_pred = (y_test_prob >= threshold).astype(int)
            else:
                y_test_pred = best_model.predict(X_test)

            score = f1_score(y_test, y_test_pred)
            report[model_name] = {"score": score, "model": best_model, "params": gs.best_params_}

            logging.info(f"{model_name} — Test F1: {score:.4f}")

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)
