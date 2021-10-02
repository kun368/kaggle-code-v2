import pandas as pd

from collections import Counter
from utils.common import compress_read, set_pandas_option

from sklearn.model_selection import train_test_split

print('------------------------------ read inputs ------------------------------')
set_pandas_option()

train = compress_read('data/train.csv')
pred = compress_read('data/test.csv')
sample_submission = compress_read('data/sample_submission.csv')

train_df, test_df = train_test_split(train, test_size=0.2, random_state=42)
train_df, test_df = pd.DataFrame(train_df), pd.DataFrame(test_df)
train_df.info()

print('------------------------------ data distribution ------------------------------')
for i in train_df.columns:
    counter = Counter(train_df[i].values)
    print(i, len(counter.keys()), counter.most_common(5))
train_df.iloc[:100].to_excel('first_100_data.xlsx')
