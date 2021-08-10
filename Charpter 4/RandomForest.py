
"""
在上一章的基础上，决策树生成过程中加入随机选取特征参数
"""
import numpy as np
from Tree import *


class RandomForest(ClassifierBase):
    _cvd_trees = {'id3': ID3Tree,
                  'c45': C45Tree,
                  'cart': CartTree}

    def __init__(self):
        super(RandomForest, self).__init__()
        self._trees = []

    @staticmethod
    def most_appearance(arr):
        u, c = np.unique(arr, return_counts=True)
        return u[np.argmax(c)]

    def fit(self, x, y, sample_weights=None, tree='cart', epoch=10, feature_bound='log', *args, **kwargs):
        x, y = np.atleast_2d(x), np.array(y)
        n_sample = len(y)
        for _ in range(epoch):
            tmp_tree = RandomForest._cvd_trees[tree]
            _indices = np.random.randint(n_sample, size=n_sample)
            if sample_weights is None:
                _local_weight = None
            else:
                _local_weight = sample_weights[_indices]
                _local_weight /= _local_weight.sum()
            tmp_tree.fit(x[_indices], y[_indices], sample_weights=_local_weight, feature_bound=feature_bound)
            self._trees.append(deepcopy(tmp_tree))

    def predict(self, x):
        _matrix = np.array([_tree.predict(x) for _tree in self._trees]).T
        return np.array([RandomForest.most_appearance(rs)] for rs in _matrix)








