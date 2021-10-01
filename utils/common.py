import pandas as pd
import os.path as path


def set_pandas_option():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 10000)
    pd.set_option('max_colwidth', 100)


def compress_read(f):
    parquet_file = f.replace('.csv', '.parquet')
    if str(f).endswith('.csv') and path.exists(f):
        pd.read_csv(f).to_parquet(parquet_file, compression='brotli')
    return pd.read_parquet(parquet_file)
