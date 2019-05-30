import numpy as np

class FullyConnect():
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.flat_input_size = np.product(input_size)
        self.output_size = output_size
        self.activation = activation
        self.init_params()

    def init_params(self):
        self.w = np.random.randn(self.output_size, self.flat_input_size) / np.sqrt(self.flat_input_size)
        self.b = np.zeros(self.output_size)

    def forward(self, x):
        self.x = x.reshape(self.flat_input_size)
        z = np.dot(self.w, self.x) + self.b
        if self.activation == 'relu':
            a = np.maximum(0, z)
        else:
            a = z
        return a

    def backward(self, delta, eta):
        dw = np.dot(delta.reshape(self.output_size, 1), self.x.reshape(1, self.flat_input_size))
        db = delta
        self.w -= eta * dw
        self.b -= eta * db

        d = np.dot(self.w.transpose(), delta)
        return d.reshape(self.input_size)

if __name__ == '__main__':
    fc = FullyConnect((16,5,5), 120)
    x = np.arange(16*5*5).reshape(16,5,5)
    y = fc.forward(x)
    print(y.shape)
    y1 = y.copy() + 1
    d = fc.backward(y1 - y, 0.1)
    print(d.shape)