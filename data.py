import numpy as np
from struct import unpack
import os

def check_dataset():
    if  not os.path.isfile('dataset/train-labels-idx1-ubyte') or \
        not os.path.isfile('dataset/train-images-idx3-ubyte') or \
        not os.path.isfile('dataset/t10k-labels-idx1-ubyte') or \
        not os.path.isfile('dataset/t10k-images-idx3-ubyte'):
        os.system('cd dataset && ./download_mnist.sh')

def read_dataset(filename):
    print("Reading %s" % filename)
    f = open(filename, "rb")
    magic, = unpack(">L", f.read(4))
    count, = unpack(">L", f.read(4))
    dtype = "images" if magic == 2051 else "labels"
    print("  Magic number: %d, %d %s!" % (magic, count, dtype))

    if dtype == "images":
        data = []
        width, = unpack(">L", f.read(4))
        height, = unpack(">L", f.read(4))
        print("  Image size: [%d, %d]" % (width, height))
        for i in range(0, 1000):#count):
            print("  Reading image: %d / %d" % (i+1, count), end="\r")
            array = [unpack("B", f.read(1))[0] for j in range(0, width*height)]
            array = np.ubyte([array[j::28] for j in range(0, 28)])
            array = (array - np.min(array)) / (np.max(array) - np.min(array))
            array = array.reshape((1,)+array.shape)
            data.append(array)
    elif dtype == "labels":
        data = [unpack("B", f.read(1))[0] for i in range(0, count)]

    f.close()
    print("")
    return (data, count)

def vectorize(num, classes):
    return [1 if i == num else 0 for i in range(classes)]

def load_mnist(train_percent=0.99):
    check_dataset()
    train_input,_ = read_dataset('dataset/train-images-idx3-ubyte')
    train_label,_ = read_dataset('dataset/train-labels-idx1-ubyte')
    test_input,_  = read_dataset('dataset/t10k-images-idx3-ubyte')
    test_label,_  = read_dataset('dataset/t10k-labels-idx1-ubyte')

    train_label = np.array([vectorize(train_label[i], 10) for i in train_label])
    test_label = np.array([vectorize(test_label[i], 10) for i in test_label])

    i = int(len(train_input)*train_percent)
    val_input = train_input[i:]
    val_label = train_label[i:]
    train_input = train_input[:i]
    train_label = train_label[:i]

    return train_input, train_label, val_input, val_label, test_input, test_label

if __name__ == '__main__':
    load_mnist()