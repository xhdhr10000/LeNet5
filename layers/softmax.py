import numpy as np

class Softmax():
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        self.x = x.reshape(self.size)
        e = np.exp(self.x - np.max(self.x))
        self.a = e / np.sum(e)
        return self.a

    def backward(self, delta):
        i = np.argmax(self.a)
        m = np.zeros((self.size, self.size))
        m[:, i] = 1
        m = np.eye(self.size) - m
        d = np.diag(self.a) - np.outer(self.a, self.a)
        d = np.dot(delta, d)
        d = np.dot(d, m)
        return d

if __name__ == '__main__':
    softmax = Softmax(3)
    x = np.arange(3)
    y = softmax.forward(x)
    print(y)
    y1 = y.copy()
    y1[2] -= 1
    d = softmax.backward(y1)
    print(d)