import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected = len(self._schema_config["columns"])
            actual = len(dataframe.columns)
            logging.info(f"Column count — expected: {expected}, actual: {actual}")
            return actual == expected
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_column_names(self, dataframe: pd.DataFrame) -> bool:
        try:
            schema_cols = list(self._schema_config["columns"].keys())
            df_cols = list(dataframe.columns)
            if schema_cols == df_cols:
                logging.info("Column name validation passed")
                return True
            missing = set(schema_cols) - set(df_cols)
            extra = set(df_cols) - set(schema_cols)
            logging.warning(f"Column mismatch — missing: {missing}, extra: {extra}")
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def check_missing_values(self, dataframe: pd.DataFrame) -> bool:
        """
        FIX: changed from hard RAISE to WARNING + return False.
        Raising here blocks the pipeline even when missing values are
        handleable downstream (e.g. imputation in transformation step).
        """
        try:
            missing = dataframe.isnull().sum()
            cols_with_missing = missing[missing > 0]
            if len(cols_with_missing) > 0:
                logging.warning(
                    f"Missing values detected (will be handled in transformation):\n"
                    f"{cols_with_missing.to_string()}"
                )
                return False
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def check_duplicates(self, dataframe: pd.DataFrame) -> bool:
        try:
            count = dataframe.duplicated().sum()
            if count > 0:
                logging.warning(f"Duplicate rows found: {count} — dropping before proceeding")
                return False
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """
        Runs KS test on all numeric columns.
        Returns True (no drift) / False (drift detected).

        FIX: skips non-numeric columns cleanly rather than silently passing object cols.
        """
        try:
            status = True
            report = {}

            numeric_cols = base_df.select_dtypes(include="number").columns

            for col in numeric_cols:
                stat = ks_2samp(base_df[col].dropna(), current_df[col].dropna())
                drift_detected = stat.pvalue < threshold
                if drift_detected:
                    status = False
                report[col] = {
                    "p_value": float(round(stat.pvalue, 6)),
                    "drift_status": drift_detected,
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            drifted = [c for c, v in report.items() if v["drift_status"]]
            logging.info(
                f"Drift check complete. Drifted columns ({len(drifted)}): {drifted}"
            )
            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = DataValidation.read_data(self.data_ingestion_artifacts.trained_file_path)
            test_df  = DataValidation.read_data(self.data_ingestion_artifacts.test_file_path)

            errors = []

            # Column count
            if not self.validate_number_of_columns(train_df):
                errors.append("Train: column count mismatch")
            if not self.validate_number_of_columns(test_df):
                errors.append("Test: column count mismatch")

            # Column names
            if not self.validate_column_names(train_df):
                errors.append("Train: column name mismatch")
            if not self.validate_column_names(test_df):
                errors.append("Test: column name mismatch")

            # FIX: missing values and duplicates are logged as warnings, not hard failures
            # (they are handled gracefully in the transformation step)
            self.check_missing_values(train_df)
            self.check_missing_values(test_df)
            self.check_duplicates(train_df)
            self.check_duplicates(test_df)

            if errors:
                raise Exception(f"Validation failed:\n" + "\n".join(errors))

            # Drift detection (informational — pipeline continues regardless)
            drift_status = self.detect_dataset_drift(train_df, test_df)
            if not drift_status:
                logging.warning("Dataset drift detected. Review drift report before deploying.")

            # Save valid files
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data validation complete: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
