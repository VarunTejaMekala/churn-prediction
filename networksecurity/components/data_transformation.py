import sys
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from networksecurity.constant.training_pipeline import TARGET_COLUMN

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):

        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    @staticmethod
    def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering for churn prediction
        """

        try:

            df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

            df["total_spend"] = df["MonthlyCharges"] * df["tenure"]

            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 60, 100],
                labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"]
            )

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:

        try:

            logging.info("Creating preprocessing pipeline")

            numerical_columns = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "charges_per_tenure",
                "total_spend"
            ]

            categorical_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "tenure_group"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:

        logging.info("Entered initiate_data_transformation method")

        try:

            logging.info("Reading train and test data")

            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )

            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # Fix TotalCharges datatype
            train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
            test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

            # Feature engineering
            train_df = self.add_engineered_features(train_df)
            test_df = self.add_engineered_features(test_df)

            logging.info("Splitting input and target features")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            target_feature_train_df = target_feature_train_df.map({"Yes": 1, "No": 0})
            target_feature_test_df = target_feature_test_df.map({"Yes": 1, "No": 0})

            logging.info("Getting preprocessing object")

            preprocessor = self.get_data_transformer_object()

            logging.info("Fitting preprocessing object")

            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(
                input_feature_train_df
            )

            transformed_input_test_feature = preprocessor_object.transform(
                input_feature_test_df
            )

            logging.info("Combining input features with target column")

            train_arr = np.c_[
                transformed_input_train_feature,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                transformed_input_test_feature,
                np.array(target_feature_test_df)
            ]

            logging.info("Saving transformed data")

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            logging.info("Saving preprocessing object")

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object
            )

            os.makedirs("final_model", exist_ok=True)

            save_object(
                "final_model/preprocessor.pkl",
                preprocessor_object
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)