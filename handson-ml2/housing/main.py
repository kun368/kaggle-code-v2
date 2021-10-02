from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils import common

print('------------------------------ read inputs ------------------------------')
common.set_pandas_option()

housing = common.compress_read('data/housing.csv')
train_df, test_df = train_test_split(housing, test_size=0.2, random_state=42)
train_df, test_df = pd.DataFrame(train_df), pd.DataFrame(test_df)

print('------------------------------ data distribution ------------------------------')
housing = train_df.copy()

for i in housing.columns:
    cnt = Counter(housing[i].values)
    print(i, len(cnt.keys()), cnt.most_common(10))
print(housing.describe().T)

housing.hist(bins=50, figsize=(20, 15))
plt.savefig('plots/hist.png', dpi=600)

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.savefig('plots/location_scatter_v1.png', dpi=600)

housing.plot(
    kind='scatter', x='longitude', y='latitude', alpha=0.4,
    s=housing['population'] / 100, label='population', figsize=(10, 7),
    c="median_house_value", cmap=plt.get_cmap('jet'), colorbar=True, linewidths=0
)
plt.savefig('plots/location_scatter_v2.png', dpi=600)

print('------------------------------ correlation ------------------------------')
corr_mat = housing.corr()
print(corr_mat['median_house_value'].sort_values(ascending=False))  # corr with label

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 12))
plt.savefig('plots/scatter_matrix.png', dpi=600)

print('------------------------------ training preparation ------------------------------')
housing = train_df.drop(columns=['median_house_value'])
housing_labels = train_df['median_house_value'].copy()

cat_features = ['ocean_proximity']
num_features = housing.drop(columns=cat_features).columns


class CombinedAttrsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(data=X, columns=num_features)
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['population_per_household'] = X['population'] / X['households']
        if self.add_bedrooms_per_room:
            X['bedrooms_per_household'] = X['total_bedrooms'] / X['households']
        return X.values


pipe = Pipeline([
    ('trans', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', verbose=1)),
            ('new_attrs', CombinedAttrsAdder()),
            ('std_scalar', StandardScaler())
        ]), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])),
    ('reg', LGBMRegressor()),
])
print(pipe.get_params())
pipe.fit(housing, housing_labels)
print([round(i, 2) for i in housing_labels[:10]])
print([round(i, 2) for i in pipe.predict(housing[:10])])
print(mean_squared_error(housing_labels, pipe.predict(housing), squared=False))

scores = cross_val_score(pipe, housing, housing_labels, scoring='neg_mean_squared_error', cv=10)
scores = np.sqrt(-scores)
print(np.mean(scores), np.std(scores), list(scores))

print('------------------------------ search parameters ------------------------------')
param_grid = [
    {'reg__n_estimators': [30, 100, 300], 'reg__max_depth': [10, 20, 30, -1]}
]
# RandomizedSearchCV
search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, return_train_score=True)
search.fit(housing, housing_labels)
print(search.best_params_)
cv_res = search.cv_results_
for mean_score, params in zip(cv_res['mean_test_score'], cv_res['params']):
    print(np.sqrt(-mean_score), params)

joblib.dump(search.best_estimator_, 'final_model.joblib', compress=3)
final_model = joblib.load('final_model.joblib')

print('------------------------------ test ------------------------------')
X_test = test_df.drop(columns=['median_house_value'])
y_test = test_df['median_house_value'].copy()
print([round(i, 2) for i in y_test[:10]])
print([round(i, 2) for i in final_model.predict(X_test[:10])])
print(mean_squared_error(y_test, final_model.predict(X_test), squared=False))
