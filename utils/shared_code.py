from collections import OrderedDict


class Summer:
    def __init__(self):
        self.sum_num = 0.0
        self.sum_weight = 0.0

    @staticmethod
    def convert(n):
        from typing import Iterable
        if isinstance(n, Iterable):
            n = next(iter(n))
        return float(n)

    def add(self, num, weight=1.0):
        self.sum_num += self.convert(num)
        self.sum_weight += self.convert(weight)

    def __str__(self):
        return "{:.4f}".format(self.sum_num / self.sum_weight)


class MultiSummer:
    def __init__(self):
        self.summers = OrderedDict()

    def put(self, key, num, weight=1.0):
        if key not in self.summers:
            self.summers[key] = Summer()
        self.summers[key].add(num, weight)

    def get(self, key):
        return self.summers.get(key, Summer())

    def __str__(self):
        return ' '.join([f'{k}: {v}' for k, v in self.summers.items()])


def set_seeds(seed=42):
    import tensorflow as tf
    import os
    import numpy as np
    import random

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def show_one_image(image):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.show()


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        import time
        self.tik = time.time()

    def stop(self):
        import time
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        import numpy as np
        return np.array(self.times).cumsum().tolist()


def to_kaggle_submission(submission, out_zip='submission.zip'):
    import pandas as pd
    import zipfile
    import os

    pd.DataFrame(submission).to_csv('submission.csv', index=False)
    with zipfile.ZipFile(out_zip, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write('submission.csv', arcname='submission.csv')
    os.remove('submission.csv')
