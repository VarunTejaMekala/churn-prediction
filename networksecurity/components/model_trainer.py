import os
import sys
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import ChurnModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_report import get_classification_score
from networksecurity.constant.training_pipeline import (
    CLASSIFICATION_THRESHOLD,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD,
)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, run_name: str, model, metric_artifact, best_params: dict):
        """
        Logs a single MLflow run.

        FIX: was called twice with the same model (once for train, once for test)
             inside the same outer block, causing two overlapping active runs.
             Now takes a run_name so train and test are logged in separate named runs.
        FIX: logs roc_auc_score (was missing).
        """
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("Churn Prediction")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_metric("f1_score",        metric_artifact.f1_score)
                mlflow.log_metric("precision_score",  metric_artifact.precision_score)
                mlflow.log_metric("recall_score",     metric_artifact.recall_score)
                mlflow.log_metric("roc_auc_score",    metric_artifact.roc_auc_score)  # FIX: added
                mlflow.log_metric("threshold",        CLASSIFICATION_THRESHOLD)

                if best_params:
                    mlflow.log_params(best_params)

                if "XGB" in type(model).__name__:
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            # MLflow logging is non-critical — log warning and continue
            logging.warning(f"MLflow tracking failed (non-fatal): {e}")

    def _check_overfitting(
        self, train_f1: float, test_f1: float, model_name: str
    ) -> None:
        """
        FIX: overfitting/underfitting check was defined in constants but never applied.
             Added explicit check with warning log.
        """
        diff = abs(train_f1 - test_f1)
        threshold = MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        if diff > threshold:
            logging.warning(
                f"[{model_name}] Possible overfitting — "
                f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, "
                f"Diff: {diff:.4f} > threshold: {threshold}"
            )
        else:
            logging.info(
                f"[{model_name}] Overfitting check passed — diff: {diff:.4f}"
            )

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        try:
            # -------------------------------------------------------
            # SMOTE — oversample minority class on train only
            # -------------------------------------------------------
            logging.info("Applying SMOTE to balance training dataset")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"Post-SMOTE shapes — X: {X_train.shape}, y: {y_train.shape}")

            # -------------------------------------------------------
            # Model candidates
            # -------------------------------------------------------
            models = {
                "Random Forest": RandomForestClassifier(verbose=0, random_state=42),
                "XGBoost":       XGBClassifier(verbosity=0, eval_metric="logloss", random_state=42),
                "AdaBoost":      AdaBoostClassifier(random_state=42),
            }

            params = {
                "Random Forest": {
                    "n_estimators":     [200, 400],
                    "max_depth":        [10, 20],
                    "min_samples_split":[2, 5],
                    "min_samples_leaf": [1, 2],
                },
                "XGBoost": {
                    "n_estimators":      [200, 400],
                    "learning_rate":     [0.05, 0.1],
                    "max_depth":         [3, 5],
                    "subsample":         [0.8, 1.0],
                    "colsample_bytree":  [0.8, 1.0],
                },
                "AdaBoost": {
                    "n_estimators":  [100, 200],
                    "learning_rate": [0.05, 0.1],
                },
            }

            # -------------------------------------------------------
            # FIX: evaluate_models now returns dict of
            #      {name: {"score": f1, "model": fitted_model, "params": best_params}}
            #      instead of {name: r2_score_float}
            # -------------------------------------------------------
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
                threshold=CLASSIFICATION_THRESHOLD,
            )

            best_model_name = max(model_report, key=lambda k: model_report[k]["score"])
            best_model_score = model_report[best_model_name]["score"]
            best_model = model_report[best_model_name]["model"]
            best_params = model_report[best_model_name]["params"]

            logging.info(
                f"Best model: {best_model_name} — Test F1: {best_model_score:.4f}"
            )

            # FIX: check expected accuracy threshold
            if best_model_score < MODEL_TRAINER_EXPECTED_SCORE:
                raise Exception(
                    f"No model met expected F1 >= {MODEL_TRAINER_EXPECTED_SCORE}. "
                    f"Best was {best_model_name} with F1={best_model_score:.4f}"
                )

            # Re-fit best model on full (SMOTE) train set
            best_model.fit(X_train, y_train)

            # -------------------------------------------------------
            # Evaluate with churn-tuned threshold
            # -------------------------------------------------------
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            y_test_prob  = best_model.predict_proba(X_test)[:, 1]

            y_train_pred = (y_train_prob >= CLASSIFICATION_THRESHOLD).astype(int)
            y_test_pred  = (y_test_prob  >= CLASSIFICATION_THRESHOLD).astype(int)

            train_metric = get_classification_score(y_train, y_train_pred, y_train_prob)
            test_metric  = get_classification_score(y_test,  y_test_pred,  y_test_prob)

            logging.info(f"Train — F1: {train_metric.f1_score:.4f} | ROC-AUC: {train_metric.roc_auc_score:.4f}")
            logging.info(f"Test  — F1: {test_metric.f1_score:.4f}  | ROC-AUC: {test_metric.roc_auc_score:.4f}")

            # Overfitting check
            self._check_overfitting(train_metric.f1_score, test_metric.f1_score, best_model_name)

            # Classification reports
            print(f"\nClassification Report — Train ({best_model_name})\n")
            print(classification_report(y_train, y_train_pred, target_names=["No Churn", "Churn"]))
            print(f"\nClassification Report — Test ({best_model_name})\n")
            print(classification_report(y_test, y_test_pred, target_names=["No Churn", "Churn"]))

            # -------------------------------------------------------
            # MLflow — FIX: separate run per split (no nested active runs)
            # -------------------------------------------------------
            self.track_mlflow(f"{best_model_name}_train", best_model, train_metric, best_params)
            self.track_mlflow(f"{best_model_name}_test",  best_model, test_metric,  best_params)

            # -------------------------------------------------------
            # Save model
            # -------------------------------------------------------
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )

            churn_model = ChurnModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, churn_model)
            save_object("final_model/model.pkl", best_model)

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )
            logging.info(f"Model trainer artifact: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
