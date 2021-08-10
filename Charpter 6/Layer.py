import numpy as np


class Layer:
    """
    self.shape:记录上个Layer和当前Layer的神经元的个数
    self.shape[0]:上个Layer所含神经元的个数
    self.shape[1]:当前Layer所含神经元的个数
    """
    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    def _activate(self, x):
        pass

    def derivative(self, y):
        pass

    def avtivate(self, x, w, bias):
        return self._activate(x.dot(w)+bias)

    def bp(self, y, w, prev_delta):
        return prev_delta.dot(w.T) * self.derivative(y)


class Sigmoid(Layer):
    def _activate(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, y):
        return y * (1 - y)


class CostLayer(Layer):
    """
    self._available_cost_functions:记录所有损失函数的字典
    self._available_transform_functions:记录所有特殊变换函数的字典
    self._cost_function、self._cost_function_name:损失函数及其名字
    self._transform_function、self._transform:记录特殊变换函数及其名字
    """
    def __init__(self, shape, cost_function='MSE', transform=None):
        super(CostLayer, self).__init__(shape)
        self._available_cost_functions = {'MSE': CostLayer._mse,
                                          'CrossEntropy': CostLayer._cross_entropy}
        self._available_transfrom_functions = {'Softmax': CostLayer._softmax,
                                               'Sigmoid': CostLayer._sigmoid}
        self._cost_function_name = cost_function
        self._cost_function = self._available_cost_functions[cost_function]
        if transform is None and cost_function == 'CrossEntropy':
            self._transform = 'Softmax'
            self._tansform_function = CostLayer._softmax
        else:
            self._transform = transform
            self._transform_function = self._available_transfrom_functions.get(transform, None)

    def __str__(self):
        return self._cost_function_name

    def _activate(self, x):
        if self._tansform_function is None:
            return x
        return self._tansform_function(x)

    def derivative(self, y, delta=None):
        pass

    @staticmethod
    def safe_exp(x):
        return np.exp(x - np.max(x, axis=1, keepdims=True))

    @staticmethod
    def _softmax(y, diff=False):
        if diff:
            return y * (1 - y)
        exp_y = CostLayer.safe_exp(y)
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(y, diff):
        if diff:
            return y * (1 - y)
        return 1/(1 + np.exp(-y))

    # 定义计算整合梯度的方法，注意这里返回的是负梯度
    def bp_first(self, y, y_pred):
        if self._cost_function_name == 'CrossEntropy' and (
                self._transform == 'Softmax' or self._transform == 'Sigmoid'):
            return y - y_pred
        dy = -self._cost_function(y, y_pred)
        if self._transform_function is None:
            return dy
        return dy * self._transform_function(y_pred, diff=True)

    @property
    def calculate(self):
        return lambda y, y_pred: self._cost_function(y, y_pred, False)

    @staticmethod
    def _mse(y, y_pred, diff=True):
        if diff:
            return -y + y_pred
        return 0.5 * np.average((y - y_pred) ** 2)

    @staticmethod
    def _cross_entropy(y, y_pred, diff=True, eps=1e-8):
        if diff:
            return -y/(y_pred + eps) + (1 - y) / (1 - y_pred + eps)
        return np.average(-y * np.log(y_pred + eps) - (1 - y) * np.log(1 - y_pred + eps))


