import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def ReLU(z):
    """The Rectified Linear Unit."""
    return np.clip(z, 0, None)

def ReLU_prime(z):
    """Derivaive of the ReLU function"""
    return (z>0).astype(np.int32)

def LReLU(z):
    return np.clip(z,0.1*z,None)

def LReLU_prime(z):
    return 0.1 + 0.9*(z>0).astype(np.int32)

class Network(object):

    def __init__(self, sizes, activation_mode):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        activation_function_dict = dict(zip(['sigmoid','ReLU','LReLU'],[sigmoid,ReLU,LReLU]))
        activation_derivative_dict = dict(zip(['sigmoid','ReLU','LReLU'],[sigmoid_prime,ReLU_prime,LReLU_prime]))
        self.activation_function = activation_function_dict[activation_mode]
        self.activation_derivative = activation_derivative_dict[activation_mode]
    
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
        for j in range(epochs):
            step = eta/(1+j//200)
            self.update_mini_batch(training_data,step)
            if j%10==0:
                print("Epoch {} Loss: {:.5f}".format(j,self.compute_cost(training_data)))
                
                

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
        delta = self.cost_derivative(activations[-1], y) * self.activation_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(),delta)*self.activation_derivative(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def compute_cost(self, training_data):
        n = len(training_data)
        total = 0
        for x,y in training_data:
            predict = self.feedforward(x)
            total += np.dot(y-predict,y-predict)/2
        return (total/n)[0][0]  

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives "partial C_x" / "partial a" for the output activations."""
        return (output_activations-y)

def generate_training_set(size):
    """
    y = sin(x_1) - cos(x_2), x_1, x_2 belongs to interval [-5, 5]
    """
    training_set = []
    for i in range(size):
        x_1 = i*10/size - 5
        for j in range(size):
            x_2 = j*10/size - 5
            y = (np.math.sin(x_1) - np.math.cos(x_2)+2)/8+0.25
            x = np.array([x_1,x_2]).reshape(2,1)
            example = tuple([x,y])
            training_set.append(example)
    return training_set

def visualization(network, training_set):
    points = []
    for x,y in training_set:
        prediction = network.feedforward(x)
        points.append([x[0],x[1],8*(y-0.25)-2,8*(float(prediction)-0.25)-2])
    points = np.array(points)
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    Z_predict = points[:,3]

    fig1 = plt.figure()
    axis1 = fig1.gca(projection='3d')
    axis1.set_xlabel('X1 Label')
    axis1.set_ylabel('X2 Label')
    axis1.set_zlabel('Y Label')
    axis1.scatter(X,Y,Z,c='b')
    fig2 = plt.figure()
    axis2 = fig2.gca(projection='3d')
    axis2.set_xlabel('X1 Label')
    axis2.set_ylabel('X2 Label')
    axis2.set_zlabel('Y Label')
    axis2.scatter(X,Y,Z_predict,c='b')
    plt.show()

if __name__ == "__main__":
    net = Network([2,8,1],'sigmoid')
    training_set = generate_training_set(40)
    net.SGD(training_set,2000,50,20)
    visualization(net,training_set)