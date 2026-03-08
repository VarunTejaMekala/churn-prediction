import pandas as pd

df = pd.read_csv("Artifacts/03_08_2026_18_01_03/data_ingestion/ingested/train.csv")

print(len(df.columns))
print(df.columns)