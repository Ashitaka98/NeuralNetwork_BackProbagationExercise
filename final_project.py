import numpy as np
import matplotlib as plt

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


class MyNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying 
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta" is the 
        learning rate 
        """
        nabla_b = []