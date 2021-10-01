from collections import Counter

import matplotlib.pyplot as plt

from utils import common

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

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.savefig('plots/location_scatter.svg')
