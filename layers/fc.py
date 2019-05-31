import numpy as np

class FullyConnect():
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.flat_input_size = np.product(input_size)
        self.output_size = output_size
        self.activation = activation
        self.init_params()

    def init_params(self):
        self.w = np.random.randn(self.flat_input_size, self.output_size) / np.sqrt(self.flat_input_size)
        # self.w = np.ones((self.output_size, self.flat_input_size))
        self.b = np.zeros(self.output_size)

    def forward(self, x):
        self.x = x.reshape(self.flat_input_size)
        z = np.dot(self.x, self.w) + self.b
        if self.activation == 'relu':
            a = np.maximum(0, z)
        else:
            a = z
        return a

    def backward(self, delta, eta):
        dw = np.outer(self.x, delta)
        db = delta
        d = np.dot(delta, self.w.transpose())

        self.w -= eta * dw
        self.b -= eta * db
        return d.reshape(self.input_size)

if __name__ == '__main__':
    fc = FullyConnect((3,3,3), 120)
    x = np.arange(27).reshape(3,3,3)
    y = fc.forward(x)
    print('###### x {} ######'.format(x.shape))
    print(x)
    print('###### y {} ######'.format(y.shape))
    print(y)
    y1 = y.copy() + 1
    d = fc.backward(y1 - y, 0.1)
    print('###### d {} ######'.format(d.shape))
    print(d)
    print('###### w {} ######'.format(fc.w.shape))
    print(fc.w)