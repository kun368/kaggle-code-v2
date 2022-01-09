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
