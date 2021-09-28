from collections import Counter

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('datas/train.csv')
test = pd.read_csv('datas/test.csv')

# basic information
train.info()
test.info()

print(np.any(pd.isna(train).values))
print(np.any(pd.isna(test).values))

# data distribution
for i in train.columns:
    cnt = Counter(train[i].values)
    print(i, len(cnt.keys()), cnt.most_common(10))

print(np.std(train['loss']), stats.skew(train['loss']))
print(np.std(np.log(train['loss'])), stats.skew(np.log(train['loss'])))


# training prep

def prep(df):
    id = df['id'].values
    for i in train.columns:
        if i == 'id' or i == 'loss':
            continue
        if str(i).startswith('cat'):
            df[i] = LabelEncoder().fit_transform(df[i])
    x = train.drop(columns=['id', 'loss']).values
    y = np.log(train['loss'].values)
    return id, x, y


train_id, train_x, train_y = prep(train)

# training
# model = XGBRegressor(verbosity=2)
model = LGBMRegressor(silent=False, n_estimators=300)
model.fit(train_x, train_y)

train_pred = model.predict(train_x)
print(train_y[:10])
print(train_pred[:10])
print(mean_squared_error(train_y, train_pred))
