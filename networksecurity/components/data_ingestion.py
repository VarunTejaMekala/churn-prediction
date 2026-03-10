import os
import sys
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.constant.training_pipeline import TARGET_COLUMN

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """Reads raw data from MongoDB and returns a cleaned DataFrame."""
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logging.info(f"Connecting to MongoDB: {database_name}.{collection_name}")
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"Loaded {len(df):,} records from MongoDB")
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Saves raw DataFrame to the feature store CSV."""
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Feature store saved: {feature_store_file_path}")
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Stratified train/test split to preserve churn class balance.

        FIX: added stratify=dataframe[TARGET_COLUMN] — without this, the minority
             churn class may be under-represented in one of the splits.
        """
        try:
            # Encode target temporarily for stratification
            target = dataframe[TARGET_COLUMN].map({"Yes": 1, "No": 0})

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=target,        # FIX: was missing
            )

            logging.info(
                f"Split complete — Train: {len(train_set):,} | Test: {len(test_set):,}"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Train and test CSVs saved.")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion complete: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
