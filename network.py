import numpy as np
import matplotlib as plt

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def ReLU(z):
    """The Rectified  Linear Unit."""
    return np.max(0,z)

def ReLU_prime(z):
    """Derivaive of the ReLU function"""
    return 1 if z>0 else 0

class Network(object):

    def __init__(self, sizes, activation_function):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.activation_function = activation_function
    
    def feedforward(self, a):
        """ Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        The "training_data" is a list of tuples "(x, y)" representing the training 
        inputs and the desired outputs. The other non-optional parameters are self-explanatory.
        If "test_data" is provided then the network will be evaluated against the test data
        after each epoch, and partial progress printed out. This is useful for tracking progress,
        but slows things down substantially.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying 
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta" is the 
        learning rate 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]
    
    
    def backprop(self, x, y):
        """
        Return a tuple "(nabla_b, nabla_w)" representing the gradient for 
        the cost function C_x. "nabla_b" and "nabla_w" are layer-by-layer lists
        of numpy arrays, similar to "self.biases"  and "self.weights".
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        
        # backward pass
        delta = self.cost_derivative(activation[-1], y)

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x / \partial a for the output activations."""
        return output_activations-y