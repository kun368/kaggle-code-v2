import pandas as pd
import os.path as path


def set_pandas_option():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 10000)
    pd.set_option('max_colwidth', 100)


def compress_read(f) -> pd.DataFrame:
    parquet_file = f.replace('.csv', '.parquet')
    if str(f).endswith('.csv') and path.exists(f):
        pd.read_csv(f).to_parquet(parquet_file, compression='brotli')
    return pd.read_parquet(parquet_file)


def df_statistic(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for col in df.columns:
        stats.append({
            'Feature': col,
            'Type': df[col].dtype,
            'Unique Values': df[col].nunique(),
            'Not Null Count': df.shape[0] - df[col].isnull().sum(),
            'Null Value Percentage': df[col].isnull().sum() * 100.0 / df.shape[0],
            'Biggest Category Percentage': df[col].value_counts(normalize=True, dropna=False).values[0] * 100.0
        })
    return pd.DataFrame(stats)


def df_hist(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    df.hist(bins=50, figsize=(20, 15), column=5)
    plt.savefig('plots/hist.png', dpi=600)
    plt.show()
