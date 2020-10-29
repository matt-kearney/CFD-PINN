# scipy is the libary used to read from a MATLAB file
import scipy.io as sio
# pathlib is a standard library from Python 3.4 that allows for proper path generation
from pathlib import Path
# numpy
import numpy as np
# random
import random
# matplotlib for visualization
import matplotlib
import matplotlib.pyplot as plt

# preprocessing of the script
def preprocessing():
    print("preprocessing")

# custon method to obtain initial vectors for the weights
def getIV():
    # here, we just have an RNG between the range of [-1,1]
    return 2*random.random()-1

# returns the value of the sigmoid function with z as the input
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

# UNUSED
# derivative of the sigmoid function
def der_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Neuron:

    def __init__(self, locked = False, value = 0):
        # define instance variables
        self.delta = 0
        self.weights = []
        self.value = value
        self.locked = locked

    # initialize the weights of the current neuron based on the size of the next layer
    def init_weights(self, size):
        # iterate through the size of the next layer and append a new weight to the current neuron
        for a in range(0, size):
            # the getIV function generates an IV on a given initialization function
            self.weights.append(getIV())

    # set the value of the neuron
    def setValue(self, value):
        self.value = value

    # get the nth weight of the neuron
    def get_weight(self, n):
        return self.weights[n]

class Layer:

    def __init__(self, size, layer_num, bias = True):
        # set the instance variables
        self.layer_num = layer_num
        self.size = size # this size does NOT include the bias
        self.neurons = []
        # iterate through the size of the layer and append a new neuron
        for a in range(0, size):
            self.neurons.append(Neuron())
        # add in the bais if applicable (should be every layer except output)
        if (bias == True):
            self.neurons.append(Neuron(locked = True, value = 1))

    # initializes the weights between this layer and the next one
    def init_weights(self, layer):
        # iterate through all neurons and initialize the weights between this and layer
        for a in self.neurons:
            a.init_weights(layer.size)

    # set the value of the vector to be arg
    def set_vector(self, arg):
        # iterate through the layer and set the value of each neuron to the corresponding value in the vector
        for a in range(0, len(arg)):
            self.neurons[a].setValue(arg[a])

    # returns an array which corresponds to the values of each neuron
    def get_vector(self):
        # create an empty array and iterate through the layer appending the value of each neuron
        vals = []
        for a in range(0, self.size):
            vals.append(self.neurons[a].value)
        return vals

    # returns the nth neuron in the layer
    def get_neuron(self, n):
        return self.neurons[n]

class Network:

    def __init__(self, input_size, output_size, num_layers = 2, gamma = .1, layer_size_override = 0):
        # initialize input and output layer sizes
        self.input_size = input_size
        self.output_size = output_size
        # the hidden layer size is generally the average between the input and output
        self.layer_size = int(np.round((input_size + output_size) / 2))
        # however, we can override this calculation from a function parameter
        if (layer_size_override != 0):
            self.layer_size = layer_size_override
        # the total number of layers is the number of hidden layers plus one input and one output layer
        self.num_layers = num_layers + 2
        # the Network object is defined as an array of layer objects
        self.layers = []
        # gamma is the learning rate
        self.gamma = gamma
        # we input an input layer
        self.layers.append(Layer(input_size, 0))
        # we iterate for the number of hidden layers and append
        for a in range(0, num_layers):
            self.layers.append(Layer(self.layer_size, a + 1))

        # finally, we append a new layer for the output
        # note, bias = False, since the output layer doesn't have a bias
        self.layers.append(Layer(output_size, self.num_layers - 1, bias=False))

        # We iterate through all layers except the output layer to initialize weights
        for a in range(0, self.num_layers - 1):
            self.layers[a].init_weights(self.layers[a+1])

        # couple of print statements for debugging
        #print("Network successfully created!")
        #print("Total number of layers: ", self.num_layers)

    # pass the values from pre layer to post layer
    def solve(self, pre, post):
        # this loop allows us to do the following:
        # since the nth weight of a neuron corresponds to the nth neuron in the next layer
        # we can simply iterate the post layer by index, and the pre layer by neuron
        # then at the end of the inner loop we simply use the activation function
        # this also allows us to skip calculation of the bias
        for a in range(0, post.size):
            # if the post layer neuron is a bias, then we pass
            if(post.get_neuron(a).locked):
                continue
            # we reset the value of the neuron
            post.get_neuron(a).value = 0
            # iterate through all the neurons connecting to the post neuron and add the weight * pre.value
            for pneuron in pre.neurons:
                post.get_neuron(a).value += pneuron.get_weight(a) * pneuron.value
            # we then update the post neurons value after passing it through the activation function (can be changed)
            post.get_neuron(a).value = sigmoid(post.get_neuron(a).value)

    # evaluates the network with the current weights for an input sequence of arg
    def evaluate(self, arg):
        # we upload the arg vector into the input layer of the network
        self.layers[0].set_vector(arg)
        # iterate through each layer calling the solve function on the current layer and the next layer
        for a in range(0, self.num_layers - 1):
            self.solve(self.layers[a], self.layers[a + 1])

    # returns the nth layer in the network
    def get_layer(self, n):
        return self.layers[n]

    # compute and update the network given the input of arg, against an expected output
    def compute(self, arg, exp):
        # first we check for mismatch errors
        # this is defined as the length of either arg or exp not being the same size of the input layer or output layer
        # respectively
        if (len(arg) != self.get_layer_size(0)):
            print("Input size mismatch error:")
            print("Network expected input: ", self.get_layer_size(0))
            print("Received input: ", len(arg))
            return -1
        if (len(exp) != self.get_layer_size(self.num_layers-1)):
            print("Output size mismatch error.")
            print("Network expected output: ", self.get_layer_size(self.num_layers-1))
            print("Received output: ", len(exp))
            return -1
        # we set the input layer with the input args
        self.layers[0].set_vector(arg)
        # iterate through the inner layers and call the solve function
        for a in range(0, self.num_layers - 1):
            self.solve(self.layers[a], self.layers[a+1])
        # now we back-propagate, which is solved dynamically and iteratively
        # despite having multiple inner loops, the total runtime of the algorithm is O(n), where n is the number
        # of weights in the network. I'm sure there's ways that this algorithm can be optimized, which is due to
        # my inexperience in coding, however every weight must be visited, so O(n) in minimum
        # we set the iterative layer to the output layer
        layer = self.num_layers - 1
        while(layer != -1):
            # it is an iterator used in case there are multiple output neurons
            it = 0
            # we iterate through every neuron in the current layer
            for neuron in self.get_neurons(layer):
                # reset a temp variable for calculating the delta
                delta = 0
                # if the current layer is NOT the output layer
                if (layer != self.num_layers - 1):
                    # First, we need to update the weights of all edges stemming from this neuron
                    for a in range(0, len(neuron.weights)):
                        neuron.weights[a] = neuron.weights[a] - self.gamma * self.get_neuron(layer+1, a).delta * neuron.value
                    # next, we need to update this delta of this neuron
                    # we iterate through all neurons in the next layer by index
                    for a in range(0, self.get_layer_size(layer+1)):
                        # we do the summation part of the delta formula, where the value is delta_z * weight_yz
                        delta += self.get_neuron(layer + 1, a).delta * neuron.get_weight(a)
                    # we finally multiply by the derivative of the sigmoid function
                    delta *= neuron.value * (1 - neuron.value)
                    # we set the current neurons delta value
                    neuron.delta = delta
                else:
                    # if we're in the output layer, we simply calculate the delta by single formula
                    neuron.delta = (neuron.value - exp[it]) * neuron.value * (1 - neuron.value)
                    # in this case, there are no weights to change, so we just continue
                # we increase the special iterator by 1 (for indexing purposes)
                it += 1
            # we decrement the layer and rinse and repeat
            layer -= 1

    # returns the neurons of a specific layer
    def get_neurons(self, layer):
        return self.layers[layer].neurons

    # returns the nth neuron of a specific layer
    def get_neuron(self, layer, n):
        return self.layers[layer].get_neuron(n)

    # returns the size of the nth layer
    def get_layer_size(self, n):
        return self.layers[n].size

    # returns the output layer
    def get_output_layer(self):
        return self.layers[self.num_layers - 1]

    # used for debugging
    def print_network(self):
        for a in range(0, self.num_layers):
            print("SIZE OF LAYER: ",self.layers[a].size)
            it = 0
            for n in self.layers[a].neurons:
                print("WEIGHTS FOR NEURON ",it," in layer ",a,": ",n.weights)
                it += 1

    # returns a vector of the output layer values
    def get_output(self):
        return self.layers[self.num_layers-1].get_vector()

    # prints the weights of all neurons
    def print_weights(self):
        b = 0
        for layer in self.layers:
            a = 0
            print("LAYER: ",b)
            for neuron in layer.neurons:
                print("NEURON ",a," WEIGHTS = ",neuron.weights, "   DELTA = ",neuron.delta)
                a += 1
            b += 1

    # returns the delta of the first output neuron
    def get_delta(self):
        return self.layers[self.num_layers-1].neurons[len(self.layers[self.num_layers-1].neurons)-1].delta

    # returns an array where the first index is the percent error and second index is relative error
    def accuracy(self, set):
        # initialize the counters
        sum1 = 0
        sum2 = 0
        it = 0
        # we iterate through the test set, which is a 2D array where each row is a vector of (input, output)
        for a in range(0, len(set)):
            # partition the set to the input and output variables
            inp = [set[a][0], set[a][1], set[a][2]]
            out = set[a][3]
            # evaluate the input through the current state of the network
            self.evaluate(inp)
            # get the output from the network
            act = self.get_output()[0]
            # we update the counters
            it += 1
            sum1 += (1/2)*(act - out)**2 # sum1 is chi squared test, for percent error
            sum2 += abs(out - act) / out # sum2 is the relative error for the point

        # return a vector for the two errors as averages over all points
        return [sum1 / it, sum2 / it]

# returns a vector in the sequence of [a, a+.01, a+.02, a+.03] where a = [0,.97]
# used for debugging
def random_input():
    i = random.random() * .97 + .03
    return [i - .03, i - .02, i - .01, i]

# main function used for debugging
def debug():

    # create a new network with the given variables
    my_network = Network(input_size = 3,
                         output_size= 1,
                         num_layers= 0,
                         gamma= .1)

    # n is the number of iterations
    n = 100000

    # establish some sets to be graphed for debugging purposes
    deltas = [] # list of the delta values
    diff = [] # list of the differences between calculated and accepted values
    x = [] # the x axis
    accuracy_set = [] # list of the accuracies for the network
    cost_set = [] # list of costs used for the network

    # iterate through n times, training the network over a random vector of inputs to the correct output
    for a in range(0,n):
        # generate a random vector sequence and run the compute function over it
        inp = random_input()
        my_network.compute([inp[0], inp[1], inp[2]],[inp[3]])
        # append the meta-data to the appropriate array
        x.append(a)
        deltas.append(my_network.get_delta())
        diff.append(my_network.get_output()[0] - inp[3])
        accuracy_set.append(inp)
        cost_set.append((1/2)*(diff[a])**2)

    # test the function over an "easy to calculate" vector
    my_network.evaluate([0,.01,.02])
    # set the output value to z
    z = my_network.get_output()[0]

    # print out some of the meta-data directly to the console
    print("OUTPUT VALUE: ", z)
    print("DELTA: ", my_network.get_delta())

    # get the accuracy vector and print
    accuracy = my_network.accuracy(accuracy_set)
    print("AVERAGE % ERROR: ", accuracy[0] * 100)
    print("AVERAGE RELATIVE DIFFERENCE: ", accuracy[1])

    # create a matplotlib graph to show the desired metadata
    # here, we are showing the cost set, which is the relative costs at each vertex
    fig, ax = plt.subplots()
    ax.plot(x, cost_set)
    ax.set(xlabel='iterations', ylabel='cost function',
           title='Cost Function over x Iterations')
    my_network.print_weights()
    plt.show()


# main function
def main():
    random.seed(3223)
    debug()
    #print("Hello world!")

    #data_folder = Path("C:/Users/ultim/Documents/2020/School Stuff/Research/AI/Data/")
    #file_to_open = data_folder / "Timestep_1.mat"
    #mat_contents = sio.loadmat(file_to_open)
    #mat_U = mat_contents['U']
    #mat_V = mat_contents['V']
    #mat_W = mat_contents['W']
    # print("U shape: ", mat_U.shape)
    # print("V size: ", mat_V.size)
    # print("W size: ", mat_W.size)
    # print(mat_U[0, 0, 0])

if __name__ == "__main__":
    main()
