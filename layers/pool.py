import numpy as np

class Pool():
    ''' Max pooling '''
    def __init__(self, channels, size=2, stride=2):
        self.channels = channels
        self.size = size
        self.stride = stride

    def forward(self, x):
        '''
        x: 3D (channels, w, h)
        '''
        iw,ih = x.shape[1:]
        ow,oh = iw // self.stride, ih // self.stride
        z = np.zeros((self.channels, ow, oh))
        self.pos = np.zeros((self.channels, ow, oh), dtype=np.int)
        for i in range(self.channels):
            for j in range(0, iw, self.stride):
                for k in range(0, ih, self.stride):
                    z[i, j//self.stride, k//self.stride] = np.max(x[i, j:j+self.size, k:k+self.size])
                    self.pos[i, j//self.stride, k//self.stride] = np.argmax(x[i, j:j+self.size, k:k+self.size])
        return z

    def backward(self, delta):
        '''
        delta: 3D (channels, w, h)
        '''
        iw,ih = delta.shape[1:]
        ow,oh = iw * self.stride, ih * self.stride
        d = np.zeros((self.channels, ow, oh))
        for i in range(self.channels):
            for j in range(0, iw):
                for k in range(0, ih):
                    d[i, j*self.stride+self.pos[i, j, k]//self.size, k*self.stride+self.pos[i, j, k]%self.size] = delta[i, j, k]
        return d

if __name__ == '__main__':
    pool = Pool(3)
    x = np.arange(3*28*28).reshape((3, 28, 28))
    y = pool.forward(x)
    print('###### y {} ######'.format(y.shape))
    print(y)
    d = pool.backward(y)
    print('###### d {} ######'.format(d.shape))
    print(d)
