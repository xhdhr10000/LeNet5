import numpy as np
from layers.conv import Conv
from layers.fc import FullyConnect
from layers.pool import Pool
from layers.softmax import Softmax

class LeNet5():
    def __init__(self, input_size):
        '''
        input_size: 3D (channels, w, h)
        '''
        self.input_size = input_size
        self.c1 = Conv(input_size[0], 6)
        self.s2 = Pool(6)
        self.c3 = Conv(6, 16, feature_mapping=[
            [1,1,1,0,0,0],
            [0,1,1,1,0,0],
            [0,0,1,1,1,0],
            [0,0,0,1,1,1],
            [1,0,0,0,1,1],
            [1,1,0,0,0,1],
            [1,1,1,1,0,0],
            [0,1,1,1,1,0],
            [0,0,1,1,1,1],
            [1,0,0,1,1,1],
            [1,1,0,0,1,1],
            [1,1,1,0,0,1],
            [1,1,0,1,1,0],
            [0,1,1,0,1,1],
            [1,0,1,1,0,1],
            [1,1,1,1,1,1]
        ])
        self.s4 = Pool(16)
        self.c5 = FullyConnect((16,4,4), 120)
        self.f6 = FullyConnect(120, 84)
        self.output = FullyConnect(84, 10)
        self.softmax = Softmax(10)

    def forward(self, x):
        out = self.c1.forward(x)
        out = self.s2.forward(out)
        out = self.c3.forward(out)
        out = self.s4.forward(out)
        out = self.c5.forward(out)
        out = self.f6.forward(out)
        out = self.output.forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, delta, eta):
        out = self.softmax.backward(delta)
        out = self.output.backward(out, eta)
        out = self.f6.backward(out, eta)
        out = self.c5.backward(out, eta)
        out = self.s4.backward(out)
        out = self.c3.backward(out, eta)
        out = self.s2.backward(out)
        out = self.c1.backward(out, eta)
        return out