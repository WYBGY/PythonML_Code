import math
import numpy as np


class Cluster:
    """
    self.x, self._y:记录数据集的变量
    self._counters:类别向量的计数器，记录第i类数据的个数
    self._sample_weight:记录样本权重属性
    self._con_chaos_cache, self._ent_cache, self._gini_cache:记录中间结果属性
    self._base:记录对数的底的属性
    """

    def __init__(self, x, y, sample_weight=None, base=2):
        self._x = x
        self._y = y
        if sample_weight is None:
            self._counters = np.bincount(self._y)
        else:
            self._counters = np.bincount(self._y, weights=sample_weight*(len(sample_weight)))
        self._sample_weight = sample_weight
        self._con_chaos_cache = self._ent_cache = self._gini_cache = None
        self._base = base

    def ent(self, ent=None, eps=1e-12):
        if self._ent_cache is not None and ent is None:
            return self._ent_cache
        _len = len(self._y)
        if ent is None:
            ent = self._counters

        _ent_cache = max(eps, -sum([_c/_len*math.log(_c/_len, self._base) if _c != 0 else 0 for _c in ent]))

        if ent is None:
            self._ent_cache = _ent_cache

        return _ent_cache

    def gini(self, p=None):
        if self._gini_cache is not None and p is None:
            return self._gini_cache

        if p is None:
            p = self._counters

        _gini_cache = 1 - np.sum((p / len(self._y)) ** 2)

        if p is None:
            self._gini_cache = _gini_cache

        return _gini_cache

    def con_chaos(self, idx, criterion='ent', features=None):
        if criterion == 'ent':
            _method = lambda cluster: cluster.ent()
        elif criterion == 'gini':
            _method = lambda cluster: cluster.gini()

        data = self._x[idx]
        if features is None:
            features = set(data)

        tmp_labels = [data == feature for feature in features]



