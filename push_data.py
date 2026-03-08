import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

print(MONGO_DB_URL)

import certifi
ca=certifi.where()

import numpy as np
import pandas as pd
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def csv_to_json_convertor(self,filepath):
        try:
            data=pd.read_csv(filepath)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def insert__data_mongodb(self, database, records, collection):
        try:
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tls=True,
                tlsCAFile=ca
            )

            db = self.mongo_client[database]
            col = db[collection]

            result = col.insert_many(records)
            return len(result.inserted_ids)

        except Exception as e:
            raise NetworkSecurityException(e, sys)



if __name__=="__main__":
    file_path = r"E:\codes\data_science\churn_prediction\churn_data\Telco-Customer-Churn.csv"
    database="varuntejamekala"
    collection="NetworkData"
    networkobj=NetworkDataExtract()
    record=networkobj.csv_to_json_convertor(filepath=file_path)
    no_of_records=networkobj.insert__data_mongodb(database=database,records=record,collection=collection)
    print(no_of_records)