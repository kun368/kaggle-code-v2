import pandas as pd
import os.path as path


def compress_read(f):
    parquet_file = f.replace('.csv', '.parquet')
    if str(f).endswith('.csv') and path.exists(f):
        pd.read_csv(f).to_parquet(parquet_file, compression='brotli')
    return pd.read_parquet(parquet_file)
