from collections import Counter, OrderedDict

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline

from utils import common

print('------------------------------ read inputs ------------------------------')
common.set_pandas_option()

spam_or_not_spam = common.compress_read('data/spam_or_not_spam.csv')
start_train_df, start_test_df = train_test_split(spam_or_not_spam, test_size=0.2, random_state=42)
start_train_df, start_test_df = pd.DataFrame(start_train_df), pd.DataFrame(start_test_df)
start_train_df.info()

print('------------------------------ training preparation ------------------------------')
train_df = start_train_df.copy()
train_label = train_df['label'].values
train_features = pd.DataFrame(train_df.drop(columns=['label']))


class Prep(BaseEstimator, TransformerMixin):
    def __init__(self, lower=True, replace_urls=True, remove_punctuation=True):
        self.lower = lower
        self.replace_urls = replace_urls
        self.remove_punctuation = remove_punctuation
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.lower:
            X['email'] = X['email'].apply(lambda x: str(x).lower())
        if self.replace_urls:
            X['email'] = X['email'].apply(lambda x: str(x).replace('URLs', 'URL'))
        if self.remove_punctuation:
            def remove(s):
                import string
                for i in string.punctuation:
                    s = s.replace(i, '')
                return s

            X['email'] = X['email'].apply(remove)
        return X


class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.word_count = Counter()
        self.word_id = OrderedDict()
        pass

    def fit(self, X, y=None):
        for e in pd.Series(X['email']).values:
            self.word_count.update(list(str(e).split()))
        idx = 0
        for k, _ in self.word_count.most_common(3000):
            self.word_id[k] = idx
            idx += 1
        return self

    def transform(self, X, y=None):
        def convert(s):
            ret = {}
            s = set(str(s).split())
            for w, id in self.word_id.items():
                ret[f'w_{id}'] = 1 if w in s else 0
            return ret

        return pd.DataFrame([convert(i) for i in X['email'].values])


pipe = Pipeline(steps=[
    ('prep', Prep()),
    ('adder', FeatureAdder()),
    ('model', LGBMClassifier())
])
print(pipe.get_params())
train_res = cross_val_predict(pipe, train_features, y=train_label, cv=5, verbose=4, n_jobs=-1)
print(classification_report(train_label, train_res, digits=4))

print('------------------------------ submit ------------------------------')
pipe.fit(train_features, y=train_label)

test_df = start_test_df.copy()
test_label = test_df['label'].values
test_features = pd.DataFrame(test_df.drop(columns=['label']))
test_result = pipe.predict(test_features)
print(test_label[:10])
print(test_result[:10])
print(classification_report(test_label, test_result, digits=4))
