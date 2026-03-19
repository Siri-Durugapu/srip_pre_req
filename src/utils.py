import pandas as pd
import re
import pyarrow.parquet as pq
import random

random.seed(42)

def load_data(sample_size=100000):
    parquet_file = pq.ParquetFile(r"C:\Users\Siri\Downloads\dataset_10M.parquet")
    dfs = []
    total = 0

    while total < sample_size:
        rg = random.randint(0, parquet_file.num_row_groups - 1)
        table = parquet_file.read_row_group(rg, columns=["DATA", "TOPIC"])
        df_chunk = table.to_pandas()
        dfs.append(df_chunk)
        total += len(df_chunk)

    df = pd.concat(dfs).sample(n=sample_size, random_state=42)

    counts = df["TOPIC"].value_counts()
    min_count = min(1000, counts.min())

    df = df.groupby("TOPIC").apply(
        lambda x: x.sample(n=min(len(x), min_count), random_state=42)
        ).reset_index(drop=True)

    print("\nClass Distribution:\n", df["TOPIC"].value_counts())

    return df


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    return clean_text(text)