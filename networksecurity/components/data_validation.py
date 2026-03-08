from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH

import os, sys
import pandas as pd

from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self, data_ingestion_artifacts: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):

        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

   
    # 1. Validate number of columns
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:

            expected_columns = self._schema_config["columns"]
            number_of_columns = len(expected_columns)

            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has {len(dataframe.columns)} columns")

            if len(dataframe.columns) == number_of_columns:
                return True
            return False

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    
    # 2. Validate column names
    
    def validate_column_names(self, dataframe: pd.DataFrame) -> bool:
        try:

            schema_columns = list(self._schema_config["columns"].keys())
            dataframe_columns = list(dataframe.columns)

            if schema_columns == dataframe_columns:
                logging.info("Column names validation passed")
                return True
            else:
                logging.info("Column names validation failed")
                return False

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # 3. Check missing values

    def check_missing_values(self, dataframe: pd.DataFrame) -> bool:
        try:

            missing_values = dataframe.isnull().sum()

            if missing_values.sum() > 0:
                logging.info(f"Missing values found:\n{missing_values}")
                return False

            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # 4. Check duplicate rows

    def check_duplicates(self, dataframe: pd.DataFrame) -> bool:
        try:

            duplicate_count = dataframe.duplicated().sum()

            if duplicate_count > 0:
                logging.info(f"Duplicate rows found: {duplicate_count}")
                return False

            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # 5. Dataset drift detection

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:

            status = True
            report = {}

            for column in base_df.columns:

                if base_df[column].dtype == "object":
                    continue

                df1 = base_df[column]
                df2 = current_df[column]

                test = ks_2samp(df1, df2)

                if threshold <= test.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report.update({
                    column: {
                        "p_value": float(test.pvalue),
                        "drift_status": is_found
                    }
                })

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:

        try:

            train_file_path = self.data_ingestion_artifacts.trained_file_path
            test_file_path = self.data_ingestion_artifacts.test_file_path

            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # Column count validation
            if not self.validate_number_of_columns(train_df):
                raise Exception("Train dataset column count mismatch")

            if not self.validate_number_of_columns(test_df):
                raise Exception("Test dataset column count mismatch")

            # Column name validation
            if not self.validate_column_names(train_df):
                raise Exception("Train dataset column names mismatch")

            if not self.validate_column_names(test_df):
                raise Exception("Test dataset column names mismatch")

            # Missing values check
            if not self.check_missing_values(train_df):
                raise Exception("Missing values found in train dataset")

            if not self.check_missing_values(test_df):
                raise Exception("Missing values found in test dataset")

            # Duplicate rows check
            if not self.check_duplicates(train_df):
                raise Exception("Duplicate rows found in train dataset")

            if not self.check_duplicates(test_df):
                raise Exception("Duplicate rows found in test dataset")

            # Dataset drift detection
            drift_status = self.detect_dataset_drift(
                base_df=train_df,
                current_df=test_df
            )

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True
            )

            test_df.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)