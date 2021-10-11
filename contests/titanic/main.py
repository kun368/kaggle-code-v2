from collections import Counter

from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline

from utils.common import *

print('------------------------------ read inputs ------------------------------')
set_pandas_option()

train = compress_read('data/train.csv')
pred = compress_read('data/test.csv')
sample_submission = compress_read('data/gender_submission.csv')

start_train_df, start_test_df = train_test_split(train, test_size=0.2, random_state=42)
start_train_df, start_test_df = pd.DataFrame(start_train_df), pd.DataFrame(start_test_df)
start_train_df.info()

print('------------------------------ data distribution ------------------------------')
train_df = start_train_df.copy()
train_df.to_excel('sample_data.xlsx', index=False)

for i in train_df.columns:
    cnt = Counter(train_df[i].values)
    print(i, len(cnt.keys()), cnt.most_common(10))

train_df.hist(bins=50, figsize=(20, 15))
plt.savefig('plots/hist.png', dpi=600)

corr_mat = train_df.corr()
print(corr_mat['Survived'].sort_values(ascending=False))

print('------------------------------ training preparation ------------------------------')
train_df = start_train_df.copy()
train_id = train_df['PassengerId'].values
train_label = train_df['Survived'].values
train_features = pd.DataFrame(train_df.drop(columns=['PassengerId', 'Survived']))


class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop_cols = ['Name', 'Ticket', 'Cabin']
        self.one_hot_cols = ['Sex', 'Embarked']
        self.one_hot_cats = dict()

    def fit(self, X, y=None):
        for c in self.one_hot_cols:
            X[c] = X[c].astype('category').cat.as_ordered()
            self.one_hot_cats[c] = X[c].cat.categories
        print(self.one_hot_cats)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X, copy=True)
        for c in self.one_hot_cols:
            X[c] = pd.Categorical(X[c], categories=self.one_hot_cats[c], ordered=True)
            X[c] = X[c].cat.codes
        X['AgeClass1'] = X['Age'] // 10
        X['AgeClass2'] = X['Age'] // 25
        X['FareClass'] = X['Fare'] // 30
        X.drop(columns=self.drop_cols, inplace=True)
        return X


pipe = Pipeline([
    ('feature_adder', FeatureAdder()),
    ('reg', LGBMClassifier(silent=True, n_estimators=30)),
])
print(pipe.get_params())
train_res = cross_val_predict(pipe, train_features, y=train_label, cv=3, verbose=4)
print(classification_report(train_label, train_res, digits=4))
