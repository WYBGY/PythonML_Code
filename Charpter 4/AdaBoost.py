from math import log
from MultinomalNB import MultinomialNB
from GaussianNB import GaussianNB
from Tree import *
from sklearn.naive_bayes import *
from sklearn.tree import *


class AdaBoost:
    _weak_clf = {'SKMNB': SKMultinomialNB,
                 'SKGNB': SKGaussianNB,
                 'SKTree': SKTree,
                 'MNB': MultinomialNB,
                 'GNB': GaussianNB,
                 'ID3': ID3Tree,
                 'C45': C45Tree,
                 'CART': CartTree}
    """
    self._clf: 记录分类器名称
    self._clfs: 记录弱分类器列表
    self._clfs_weights: 记录话语权列表
    """

    def __init__(self):
        self._clf, self._clfs, self._clfs_weights = "", [], []

    def fit(self, x, y, sample_weight=None, clf=None, epoch=10, eps=1e-12, **kwargs):
        if clf is None or AdaBoost._weak_clf[clf] is None:
            clf = 'CART'
            kwargs = {'max_depth': 1}
        self._clf = clf
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        else:
            sample_weight = np.array(sample_weight)

        for _ in range(epoch):
            tmp_clf = AdaBoost._weak_clf[clf](**kwargs)
            tmp_clf.fit(x, y, sample_weight)
            y_pred = tmp_clf.predict(x)
            em = min(max((y_pred != y).dot(self.sample_weight[:, None])[0], eps), 1-eps)
            am = 0.5 * log(1/em - 1)
            sample_weight *= np.exp(-am * y * y_pred)
            sample_weight /= np.sum(sample_weight)
            self._clfs.append(deepcopy(tmp_clf))
            self._clfs_weights.append(am)

    def predict(self, x):
        x = np.atleast_2d(x)
        rs = np.zeros(len(x))
        for clf, am in zip(self._clfs, self._clfs_weights):
            rs += am * clf.predict(x)
        return np.sign(rs)
    





