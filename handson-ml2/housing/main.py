from collections import Counter
import pandas as pd
from utils import common
import matplotlib.pyplot as plt

housing = common.compress_read('data/housing.csv')
housing.info()

# data distribution
for i in housing.columns:
    cnt = Counter(housing[i].values)
    print(i, len(cnt.keys()), cnt.most_common(10))

common.set_pandas_option()
print(housing.describe().T)

housing.hist(bins=50, figsize=(20, 15))
plt.savefig('plots/hist.svg')
