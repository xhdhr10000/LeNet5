import numpy as np

class Conv():
    def __init__(self, input_channels, filters, kernel=5, feature_mapping=None, activation='relu'):
        self.input_channels = input_channels
        self.filters = filters
        self.kernel = kernel
        self.feature_mapping = feature_mapping
        if self.feature_mapping is None:
            self.feature_mapping = np.ones((self.filters, self.input_channels))
        self.activation = activation
        self.init_params()

    def init_params(self):
        """
        w: 4D (filters, ic, w, h)
        b: 1D (filters)
        """
        self.w = np.random.randn(self.filters, self.input_channels, self.kernel, self.kernel) / np.sqrt(self.input_channels)
        # self.w = np.ones((self.filters, self.input_channels, self.kernel, self.kernel))
        for i in range(self.filters):
            for j in range(self.input_channels):
                if not self.feature_mapping[i][j]: self.w[i][j] = 0
        self.b = np.zeros(self.filters)

    def conv(self, x, w, mapping=None):
        """
        x,w: 3D (ic, w, h)
        output: 2D (w, h)
        """
        dim = np.subtract(x[0].shape, w[0].shape) + 1
        if mapping is None: mapping = np.ones(x.shape[0])
        a = np.zeros(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                p = np.multiply(x[:, i:i+w.shape[1], j:j+w.shape[2]], w).sum((1,2))
                a[i][j] = np.sum(p * mapping)
        return a

    def forward(self, x):
        """
        x: 3D (input_channel, w, h)
        output: 3D (filters, w, h)
        """
        self.x = x
        a = []
        for i in range(self.filters):
            self.z = self.conv(x, self.w[i], self.feature_mapping[i]) + self.b[i]
            if self.activation == 'relu':
                a.append(np.maximum(0, self.z))
            else:
                a.append(self.z)
        self.a = np.array(a)
        return self.a

    def backward(self, delta, eta):
        """
        delta: 3D (filters, w, h)
        output: 3D (input_channel, w, h)
        """
        if self.activation == 'relu':
            delta = delta * (self.z >= 0)
        d = np.pad(delta, ((0,),(self.kernel-1,),(self.kernel-1,)), mode='constant', constant_values=0)
        ds = []
        for i in range(self.input_channels):
            w = np.array([np.rot90(np.rot90(self.w[j][i])) for j in range(self.filters)])
            ds.append(self.conv(d, w))
        ds = np.array(ds)

        for i in range(self.filters):
            d = np.array([delta[i]])
            self.b[i] -= eta * np.sum(d)
            for j in range(self.input_channels):
                if not self.feature_mapping[i][j]: continue
                x = np.array([self.x[j]])
                dw = self.conv(x, d, self.feature_mapping[i][j])
                self.w[i][j] -= eta * dw

        return ds


if __name__ == '__main__':
    conv = Conv(6, 3, 3, feature_mapping=
        [[1,1,1,0,0,0],
         [0,1,1,1,0,0],
         [0,0,1,1,1,0]])
    x = np.ones((6, 6, 6)) / 27
    y = conv.forward(x)
    y1 = y.copy() + 1
    print('###### y {} ######'.format(y.shape))
    print(y)
    ds = conv.backward(y1-y, 0.1)
    print('###### delta {} ######'.format(ds.shape))
    print(ds)
    print('###### w {} ######'.format(conv.w.shape))
    print(conv.w)