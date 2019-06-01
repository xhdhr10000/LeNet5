import numpy as np
from PIL import Image
from loss import MSE, LogLikelihood
from lenet5 import LeNet5
from data import load_mnist

epochs = 3
shuffle = True
lr = 0.001

def main():
    train_input, train_label, val_input, val_label, test_input, test_label = load_mnist()
    seq = np.arange(len(train_input))

    net = LeNet5(train_input[0].shape)
    for epoch in range(epochs):
        if shuffle: np.random.shuffle(seq)
        for step in range(len(train_input)):
            i = seq[step]
            x = train_input[i]
            y_true = train_label[i]
            y = net.forward(x)
            loss = LogLikelihood.loss(y_true, y)
            dloss = LogLikelihood.derivative(y_true, y)
            print('Epoch %d step %d loss %f' % (epoch, step, loss))
            d = net.backward(dloss, lr)

            if step > 0 and step % 1000 == 0:
                correct = 0
                loss = 0
                for i in range(len(val_input)):
                    x = val_input[i]
                    y_true = val_label[i]
                    y = net.forward(x)
                    loss += LogLikelihood.loss(y_true, y)
                    if np.argmax(y) == np.argmax(y_true): correct += 1
                print('Validation accuracy: %.2f%%, average loss: %f' % (correct/len(val_input)*100, loss/len(val_input)))

if __name__ == '__main__':
    main()