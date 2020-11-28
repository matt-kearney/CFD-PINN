# module imports
import random
import numpy as np

# local imports
import nutil


# returns the value of the sigmoid function with z as the input
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


# custom method to obtain initial vectors for the weights
def get_iv():
    # here, we just have an RNG between the range of [-.5,.5]
    return 2 * random.random() - .1


class Neuron:

    def __init__(self, locked=False, value=0):
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
            self.weights.append(get_iv())

    # set the value of the neuron
    def setValue(self, value):
        self.value = value

    # get the nth weight of the neuron
    def get_weight(self, n):
        return self.weights[n]

    # sets the weights to the given array
    def set_weights(self, w):
        for weight in w:
            self.weights.append(weight)


class Layer:

    def __init__(self, size, layer_num, bias=True, load=False):
        # set the instance variables
        self.layer_num = layer_num
        self.size = size  # this size does NOT include the bias
        self.neurons = []
        # for loading
        if (load == True):
            return
        # iterate through the size of the layer and append a new neuron
        for a in range(0, size):
            self.neurons.append(Neuron())
        # add in the bais if applicable (should be every layer except output)
        if (bias == True):
            self.neurons.append(Neuron(locked=True, value=1))

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


def solve(pre, post):
    # this loop allows us to do the following:
    # since the nth weight of a neuron corresponds to the nth neuron in the next layer
    # we can simply iterate the post layer by index, and the pre layer by neuron
    # then at the end of the inner loop we simply use the activation function
    # this also allows us to skip calculation of the bias
    for a in range(0, post.size):
        # if the post layer neuron is a bias, then we pass
        if (post.get_neuron(a).locked):
            continue
        # we reset the value of the neuron
        post.get_neuron(a).value = 0
        # iterate through all the neurons connecting to the post neuron and add the weight * pre.value
        for pneuron in pre.neurons:
            post.get_neuron(a).value += pneuron.get_weight(a) * pneuron.value
        # we then update the post neurons value after passing it through the activation function (can be changed)
        post.get_neuron(a).value = sigmoid(post.get_neuron(a).value)


class Network:

    def __init__(self, input_size=0, output_size=1, num_layers=2, gamma=.1, layer_size_override=0):
        # for loading and saving
        if input_size == 0:
            return
        # initialize input and output layer sizes
        self.input_size = input_size
        self.output_size = output_size
        # the hidden layer size is generally the average between the input and output
        self.layer_size = int(np.round((input_size + output_size) / 2))
        # however, we can override this calculation from a function parameter
        if layer_size_override != 0:
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
            self.layers[a].init_weights(self.layers[a + 1])

        # couple of print statements for debugging
        # print("Network successfully created!")
        # print("Total number of layers: ", self.num_layers)

    # pass the values from pre layer to post layer

    # evaluates the network with the current weights for an input sequence of arg
    def evaluate(self, arg):
        # we upload the arg vector into the input layer of the network
        self.layers[0].set_vector(arg)
        # iterate through each layer calling the solve function on the current layer and the next layer
        for a in range(0, self.num_layers - 1):
            solve(self.layers[a], self.layers[a + 1])

    # returns the nth layer in the network
    def get_layer(self, n):
        return self.layers[n]

    # compute and update the network given the input of arg, against an expected output
    def compute(self, arg, exp):
        # first we check for mismatch errors
        # this is defined as the length of either arg or exp not being the same size of the input layer or output layer
        # respectively
        if len(arg) != self.get_layer_size(0):
            nutil.debug("Input size mismatch error:")
            nutil.debug("Network expected input: ", self.get_layer_size(0))
            nutil.debug("Received input: ", len(arg))
            return -1
        if len(exp) != self.get_layer_size(self.num_layers - 1):
            nutil.debug("Output size mismatch error.")
            nutil.debug("Network expected output: ", self.get_layer_size(self.num_layers - 1))
            nutil.debug("Received output: ", len(exp))
            return -1
        # we set the input layer with the input args
        self.layers[0].set_vector(arg)
        # iterate through the inner layers and call the solve function
        for a in range(0, self.num_layers - 1):
            solve(self.layers[a], self.layers[a + 1])
        # now we back-propagate, which is solved dynamically and iteratively
        # despite having multiple inner loops, the total runtime of the algorithm is O(n), where n is the number
        # of weights in the network. I'm sure there's ways that this algorithm can be optimized, which is due to
        # my inexperience in coding, however every weight must be visited, so O(n) in minimum
        # we set the iterative layer to the output layer
        layer = self.num_layers - 1
        while layer != -1:
            # it is an iterator used in case there are multiple output neurons
            it = 0
            # we iterate through every neuron in the current layer
            for neuron in self.get_neurons(layer):
                # reset a temp variable for calculating the delta
                delta = 0
                # if the current layer is NOT the output layer
                if layer != self.num_layers - 1:
                    # First, we need to update the weights of all edges stemming from this neuron
                    for a in range(0, len(neuron.weights)):
                        neuron.weights[a] = neuron.weights[a] - self.gamma * self.get_neuron(layer + 1,
                                                                                             a).delta * neuron.value
                    # next, we need to update this delta of this neuron
                    # we iterate through all neurons in the next layer by index
                    for a in range(0, self.get_layer_size(layer + 1)):
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
            nutil.debug("SIZE OF LAYER: ", self.layers[a].size)
            it = 0
            for n in self.layers[a].neurons:
                msg = "WEIGHTS FOR NEURON " + str(it) + " in layer " + str(a) + ": " + str(n.weights)
                nutil.debug(msg)
                it += 1

    # returns a vector of the output layer values
    def get_output(self):
        return self.layers[self.num_layers - 1].get_vector()

    # prints the weights of all neurons
    def print_weights(self):
        b = 0
        for layer in self.layers:
            a = 0
            nutil.debug("LAYER: ", b)
            for neuron in layer.neurons:
                msg = "NEURON " + str(a) + " WEIGHTS = " + str(neuron.weights) + "   DELTA = " + str(neuron.delta)
                nutil.debug(msg)
                a += 1
            b += 1

    # returns the delta of the first output neuron
    def get_delta(self):
        return self.layers[self.num_layers - 1].neurons[len(self.layers[self.num_layers - 1].neurons) - 1].delta

    # returns an array where the first index is the percent error and second index is relative error
    def accuracy(self, inp, out):
        # initialize the counters
        sum1 = 0
        sum2 = 0
        it = 0
        # we iterate through the test set, which is a 2D array where each row is a vector of (input, output)
        for a in range(0, len(inp)):
            # evaluate the input through the current state of the network
            self.evaluate(inp[a])
            # get the output from the network
            act = self.get_output()[0]
            # we update the counters
            it += 1
            sum1 += (1 / 2) * (act - out[a]) ** 2  # sum1 is chi squared test, for percent error
            sum2 += abs(out[a] - act) / out[a]  # sum2 is the relative error for the point

        # return a vector for the two errors as averages over all points
        return [sum1 / it, sum2 / it]

    # returns an array of basic network info (input size, num layers, layer size, gamma)
    def get_info(self):
        return self.input_size, self.num_layers, self.layer_size, self.gamma
