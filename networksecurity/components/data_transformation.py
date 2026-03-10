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
    DataValidationArtifact,
)
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Domain-specific feature engineering for churn prediction.

        FIX: tenure_group used pd.Categorical — OneHotEncoder needs string dtype.
             Converted to str explicitly to avoid silent encoding failures.
        FIX: added service_count feature (strong churn predictor).
        FIX: guarded division by zero with + 1 instead of raw division.
        """
        try:
            df = df.copy()

            # Spend intensity per month
            df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

            # Lifetime spend
            df["total_spend"] = df["MonthlyCharges"] * df["tenure"]

            # Number of active add-on services (strong churn signal)
            service_cols = [
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
            ]
            active_cols = [c for c in service_cols if c in df.columns]
            if active_cols:
                df["service_count"] = df[active_cols].apply(
                    lambda row: sum(
                        1 for v in row
                        if str(v).lower() not in ["no", "no internet service"]
                    ),
                    axis=1,
                )

            # Tenure bucket — FIX: cast to str so OHE handles it correctly
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 60, 100],
                labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"],
            ).astype(str)   # FIX: was pd.Categorical — breaks OneHotEncoder

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> ColumnTransformer:
        try:
            logging.info("Building preprocessing ColumnTransformer")

            numerical_columns = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "charges_per_tenure",
                "total_spend",
                "service_count",        # FIX: added new engineered feature
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
                "tenure_group",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation")
        try:
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # Fix TotalCharges dtype (blank strings in Telco dataset)
            for df in [train_df, test_df]:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

            # Feature engineering
            train_df = self.add_engineered_features(train_df)
            test_df  = self.add_engineered_features(test_df)

            # Split features / target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            input_feature_test_df  = test_df.drop(columns=[TARGET_COLUMN])

            target_feature_train = train_df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
            target_feature_test  = test_df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

            # Fit preprocessor on train only
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_train = preprocessor_object.transform(input_feature_train_df)
            transformed_test  = preprocessor_object.transform(input_feature_test_df)

            # Stack features + target into final arrays
            train_arr = np.c_[transformed_train, np.array(target_feature_train)]
            test_arr  = np.c_[transformed_test,  np.array(target_feature_test)]

            # Persist
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, test_arr
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object,
            )

            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation complete: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
