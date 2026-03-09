import os
import sys
import mlflow

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_report import (
    get_classification_score
)
from imblearn.over_sampling import SMOTE


import mlflow
import mlflow.sklearn
import mlflow.xgboost

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

    def track_mlflow(self, best_model, classificationmetric, params):


        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Churn prediction")

        with mlflow.start_run():

            # Metrics
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            # Log Parameters
            if params is not None:
                mlflow.log_params(params)

            # Log Model
            if "XGB" in str(type(best_model)) or "XGBClassifier" in str(type(best_model)):
                mlflow.xgboost.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            logging.info("Applying SMOTE to balance training dataset")

            smote = SMOTE(random_state=42)

            X_train, y_train = smote.fit_resample(X_train, y_train)

            logging.info(f"After SMOTE X_train shape: {X_train.shape}")
            logging.info(f"After SMOTE y_train shape: {y_train.shape}")
        
            models = {
                "Random Forest": RandomForestClassifier(verbose=0),
                "XGBoost": XGBClassifier(verbosity=0, eval_metric="logloss"),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {

                 "Random Forest": {
                    "n_estimators": [200, 400],
                    "max_depth": [10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "XGBoost": {
                    "n_estimators": [200, 400],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },

                "AdaBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name}")

            best_model.fit(X_train, y_train)

            # Probability predictions
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            y_test_prob = best_model.predict_proba(x_test)[:, 1]

            threshold = 0.35

            y_train_pred = (y_train_prob > threshold).astype(int)
            y_test_pred = (y_test_prob > threshold).astype(int)

            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred
            )
            
            self.track_mlflow(best_model,classification_train_metric,params)

            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred
            )

            self.track_mlflow(best_model,classification_test_metric,params)

            print("\nClassification Report (Train Data)\n")
            print(classification_report(y_train, y_train_pred))

            print("\nClassification Report (Test Data)\n")
            print(classification_report(y_test, y_test_pred))

            logging.info(f"Train Metrics: {classification_train_metric}")
            logging.info(f"Test Metrics: {classification_test_metric}")

            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )

            save_object("final_model/model.pkl", best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model_trainer_artifact = self.train_model(
                X_train,
                y_train,
                x_test,
                y_test
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)