import numpy as np

class Softmax():
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        self.x = x.reshape(self.size)
        self.a = np.exp(self.x) / np.sum(np.exp(self.x))
        return self.a

    def backward(self, delta):
        return delta * (self.a - np.square(self.a))

if __name__ == '__main__':
    softmax = Softmax(10)
    x = np.arange(10)
    y = softmax.forward(x)
    print(y)
    y1 = y.copy() + 1
    d = softmax.backward(y1-y)
    print(d)