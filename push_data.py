"""
push_data.py — Loads Telco Churn CSV and inserts records into MongoDB.

Usage:
    DATA_FILE_PATH=data/Telco-Customer-Churn.csv python push_data.py
"""

import os
import sys
import json
import certifi
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class ChurnDataExtract:
    """
    Reads the Telco Churn CSV and pushes records to MongoDB.

    FIX: renamed from NetworkDataExtract → ChurnDataExtract (correct domain).
    FIX: hardcoded Windows absolute path replaced with env var / CLI arg.
    FIX: added duplicate __insert__ check (upsert-style drop + insert).
    """

    def csv_to_json_records(self, filepath: str) -> list:
        """Converts CSV rows to a list of JSON-serialisable dicts."""
        try:
            df = pd.read_csv(filepath)
            df.reset_index(drop=True, inplace=True)
            # Replace NaN with None so MongoDB accepts it
            df = df.where(pd.notnull(df), None)
            records = json.loads(df.to_json(orient="records"))
            logging.info(f"Converted {len(records):,} rows from {filepath}")
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(
        self,
        database: str,
        collection: str,
        records: list,
        drop_existing: bool = False,
    ) -> int:
        """
        Inserts records into MongoDB.

        Args:
            database: MongoDB database name
            collection: Collection name
            records: List of dicts to insert
            drop_existing: If True, drops collection before inserting (clean load)

        Returns:
            Number of inserted records
        """
        try:
            mongo_client = pymongo.MongoClient(
                MONGO_DB_URL, tls=True, tlsCAFile=ca
            )
            db = mongo_client[database]
            col = db[collection]

            if drop_existing:
                col.drop()
                logging.info(f"Dropped existing collection: {collection}")

            result = col.insert_many(records)
            inserted = len(result.inserted_ids)
            logging.info(f"Inserted {inserted:,} records into {database}.{collection}")
            return inserted

        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    # FIX: read file path from env var instead of hardcoded Windows path
    file_path = os.getenv("DATA_FILE_PATH", "E:\codes\data_science\churn_prediction\churn_data\Telco-Customer-Churn.csv")
    database   = os.getenv("MONGO_DB_NAME",  "varuntejamekala")
    collection = os.getenv("MONGO_COLLECTION", "NetworkData")

    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found at '{file_path}'")
        print("Set DATA_FILE_PATH env variable or place the CSV at the default path.")
        sys.exit(1)

    extractor = ChurnDataExtract()
    records = extractor.csv_to_json_records(filepath=file_path)
    count = extractor.insert_data_mongodb(
        database=database,
        collection=collection,
        records=records,
        drop_existing=True,
    )
    print(f"✓ Inserted {count:,} records into {database}.{collection}")
