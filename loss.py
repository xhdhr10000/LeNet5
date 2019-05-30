import numpy as np

class MSE():
    def loss(y_true, y_pred):
        return np.sum(np.square(y_true - y_pred)) / 2.

    def derivative(y_true, y_pred):
        return y_pred - y_true