from Layer import *
from Optimizers import *
import numpy as np


class NaiveNN(ClassifierBase):
    def __init__(self):
        super(NaiveNN, self).__init__()
        self._layers, self._weights, self._bias = [], [], []
        self._w_optimizer = self._b_optimizer = None
        self._current_dimension = 0

    def add(self, layer):
        if not self._layers:
            self._layers, self._current_dimension = [layer], layer.shape[1]
            self._add_params(layer.shape)
        else:
            _next = layer.shape[0]
            layer.shape = (self._current_dimension, _next)
            self._add_layer(layer, self._current_dimension, _next)

    def _add_params(self, shape):
        self._weights.append(np.random.randn(*shape))
        self._bias.append(np.zeros((1, shape[1])))

    def _add_layer(self, layer, *args):
        _current, _next = args
        self._add_params((_current, _next))
        self._current_dimension = _next
        self._layers.append(layer)

    def fit(self, x, y, lr=0.001, optimizer='Adam', epoch=10, batch_size=0, train_rate=None):
        self._init_optimizers(optimizer, lr, epoch)
        layer_width = len(self._layers)
        if train_rate is not None:
            train_rate = float(train_rate)
            train_len = int(len(x) * train_rate)
            shuffle_suffix = np.random.permutation(len(x))
            x, y = x[shuffle_suffix], y[shuffle_suffix]
            x_train, y_train = x[:train_len], y[:train_len]
            x_test, y_test = x[train_len:], y[train_len:]
        else:
            x_train = x_test = x
            y_train = y_test = y
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len >= batch_size
        train_repeat = int(train_len/batch_size) + 1

        for counter in range(epoch):
            for _ in range(train_repeat):
                if do_random_batch:
                    batch = np.random.choice(train_len, batch_size)
                    x_batch, y_batch = x_train[batch], y_train[batch]
                else:
                    x_batch, y_batch = x_train, y_train
                self._w_optimizer.update()
                self._b_optimizer.update()
                _activations = self._get_activations(x_batch)
                _deltas = [self._layers[-1].bp_first(y_batch, _activations[-1])]
                for i in range(-1, -len(_activations), -1):
                    _deltas.append(self._layers[i-1].bp(_activations[i-1], self._weights, _deltas[-1]))

                for i in range(layer_width-1, 0, -1):
                    self._opt(i, _activations[i-1], _deltas[layer_width-i-1])
                self._opt(0, x_train, _deltas[-1])

    def _init_optimizers(self, optimizer, lr, epoch):
        _opt_fac = OptFactory()
        self._w_optimizer = _opt_fac.get_optimizer_by_name(optimizer, self._weights[:-1], lr, epoch)
        self._b_optimizer = _opt_fac.get_optimizer_by_name(optimizer, self._bias[:-1], lr, epoch)

    def _get_activations(self, x):
        _activations = [self._layers[0].avtivate(x, self._weights[0], self._bias[0])]
        for i, layer in enumerate(self._layers[1:]):
            _activations.append(layer.avtivate(_activations[-1], self._weights[i+1], self._bias[i+1]))
        return _activations

    def _opt(self, i, _activation, _delta):
        self._weights[i] += self._w_optimizer.run(i, _activation.T.dot(_delta))
        self._bias[i] += self._b_optimizer.run(i, np.sum(_delta, axis=0, keepdims=True))

    def predict(self, x, get_raw_results=False):
        y_pred = self._get_prediction(np.atleast_2d(x))
        if get_raw_results:
            return y_pred
        return np.argmax(y_pred, axis=1)

    # def _get_prediction(self, x):
    #     return self._get_activations(x)[-1]

    def _get_prediction(self, x, batch_size=1e6):
        single_batch = int(batch_size/np.prod(x.shape[1:]))
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            return self._get_activations(x).pop()
        epoch = int(len(x)/single_batch)
        if not len(x)%single_batch:
            epoch += 1
        rs, count = [self._get_activations(x[:single_batch]).pop()], single_batch
        while count < len(x):
            count += single_batch
            if count >= len(x):
                rs.append(self._get_activations(x[single_batch:]).pop())
            else:
                rs.append(self._get_activations(x[count - single_batch:count]).pop())
        return np.vstack(rs)














