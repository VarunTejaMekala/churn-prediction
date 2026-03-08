import os
import sys

from imblearn.over_sampling import SMOTE
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
)

from networksecurity.utils.ml_utils.metric.classification_report import (
    get_classification_score,
)

from sklearn.ensemble import RandomForestClassifier


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


    def train_model(self, X_train, y_train, x_test, y_test):

        try:

            logging.info("Applying SMOTE to balance training dataset")

            smote = SMOTE(random_state=42)

            X_train, y_train = smote.fit_resample(X_train, y_train)

            logging.info(f"After SMOTE X_train shape: {X_train.shape}")
            logging.info(f"After SMOTE y_train shape: {y_train.shape}")


            # Final tuned Random Forest model
            best_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                criterion="gini",
                max_features="log2",
                random_state=42,
                n_jobs=-1
            )

            logging.info("Training Random Forest model")

            best_model.fit(X_train, y_train)


            # Prediction probabilities
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            y_test_prob = best_model.predict_proba(x_test)[:, 1]

            # Threshold tuning
            threshold = 0.41

            y_train_pred = (y_train_prob > threshold).astype(int)
            y_test_pred = (y_test_prob > threshold).astype(int)


            # Metrics
            # Metrics
            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred
            )

            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred
            )

            print("\nClassification Report (Train Data)\n")
            print(classification_report(y_train, y_train_pred))

            print("\nClassification Report (Test Data)\n")
            print(classification_report(y_test, y_test_pred))

            logging.info(f"Train Metrics: {classification_train_metric}")
            logging.info(f"Test Metrics: {classification_test_metric}")


            # Load preprocessor
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )


            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(model_dir_path, exist_ok=True)


            # Combine preprocessing + model
            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model
            )


            # Save pipeline model
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )


            os.makedirs("final_model", exist_ok=True)


            # Save deployment model
            save_object(
                file_path="final_model/model.pkl",
                obj=best_model
            )


            # Artifact creation
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )


            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:

            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )


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
                y_test,
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)