import numpy as np
from PIL import Image
from loss import MSE
from lenet5 import LeNet5
from data import load_mnist

epochs = 30
shuffle = True
lr = 0.01

def main():
    train_input, train_label, val_input, val_label, test_input, test_label = load_mnist()
    seq = np.arange(len(train_input))
    if shuffle: np.random.shuffle(seq)

    net = LeNet5(train_input[0].shape)
    for epoch in range(epochs):
        for step in range(len(train_input)):
            i = seq[step]
            x = train_input[i]
            y_true = train_label[i]
            y = net.forward(x)
            loss = MSE.loss(y_true, y)
            dloss = MSE.derivative(y_true, y)
            print('Epoch %d step %d loss %f' % (epoch, step, loss))
            d = net.backward(dloss, lr)

if __name__ == '__main__':
    main()