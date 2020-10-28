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

def preprocessing():
    print("preprocessing")

# custon method to obtain initial vectors for the weights
def getIV():
    return random.random()*2 - 1

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

class Neuron:

    def __init__(self, locked = False, value = 0):
        self.delta = 0
        self.weights = []
        self.value = value
        self.locked = locked

    def init_weights(self, size):
        for a in range(0, size):
            self.weights.append(getIV())

    def setValue(self, value):
        self.value = value

class Layer:

    def __init__(self, size, layer_num, bias = True):
        self.layer_num = layer_num
        self.size = size # this size does NOT include the bias
        self.neurons = []
        for a in range(0, size):
            self.neurons.append(Neuron())
        # add in the bais if applicable
        if (bias == True):
            self.neurons.append(Neuron(locked = True, value = 1))

    # initializes the weights between this layer and the next one
    def init_weights(self, layer, out_layer_num):
        for a in self.neurons:
            if(layer.layer_num == out_layer_num):
                a.init_weights(layer.size)
            else:
                a.init_weights(layer.size + 1)

    # set the value of the vector to be arg
    def set_vector(self, arg):
        for a in range(0, len(arg)):
            self.neurons[a].setValue(arg[a])

    def get_vector(self):
        vals = []
        for a in range(0, self.size):
            vals.append(self.neurons[a].value)
        return vals



class Network:

    # In our case, output_size will always be 1, so layer_size = input_size / 2
    def __init__(self, input_size, output_size, num_layers = 2, gamma = .1, layer_size_override = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = int(np.round((input_size + output_size) / 2))
        if (layer_size_override != 0):
            self.layer_size = layer_size_override
        self.num_layers = num_layers + 2
        self.layers = []
        self.gamma = gamma
        self.layers.append(Layer(input_size, 0))
        for a in range(0, num_layers):
            self.layers.append(Layer(self.layer_size, a + 1))
        self.layers.append(Layer(output_size, self.num_layers - 1, bias=False))
        for a in range(0, self.num_layers - 1):
            self.layers[a].init_weights(self.layers[a+1], self.num_layers-1)
        print("Network successfully created!")
        print("Total number of layers: ", self.num_layers)


    # pass the values from pre layer to post layer
    def solve(self, pre, post, layer):
        # this loop allows us to do the following:
        # since the nth weight of a neuron corresponds to the nth neuron in the next layer
        # we can simply iterate the post layer by index, and the pre layer by neuron
        # then at the end of the inner loop we simply use the activation function
        # this also allows us to skip calculation of the bias
        for a in range(0, post.size):
            if(post.neurons[a].locked):
                continue
            post.neurons[a].value = 0
            for pneuron in pre.neurons:
                post.neurons[a].value += pneuron.weights[a] * pneuron.value
            post.neurons[a].value = sigmoid(post.neurons[a].value)

    # determines the error between the calculated output and the expected output
    def error(self, calc, exp):
        sum = 0
        for a in range(0, len(calc)):
            sum += (calc[a] - exp[a])**2
        return sum

    # evaluates the network with the current weights for an input sequence of arg
    def evalutate(self, arg):
        self.layers[0].set_vector(arg)
        for a in range(0, self.num_layers - 1):
            self.solve(self.layers[a], self.layers[a + 1], a)
            #self.solve(self.layers[a], self.layers[a + 1], a) # for debugging


    # compute and update the network given the input of arg, against an expected output
    def compute(self, arg, exp):
        # first we check for mismatch errors
        if (len(arg) != self.layers[0].size):
            print("Input size mismatch error:")
            print("Network expected input: ", self.layers[0].size)
            print("Received input: ", len(arg))
            return -1
        if (len(exp) != self.layers[self.num_layers-1].size):
            print("Output size mismatch error.")
            print("Network expected output: ", self.layers[self.num_layers-1].size)
            print("Received output: ", len(exp))
            return -1
        # we set the input layer with the input args
        self.layers[0].set_vector(arg)
        # iterate through the inner layers and solve
        for a in range(0, self.num_layers - 1):
            self.solve(self.layers[a], self.layers[a+1], a)
        # now we backpropagate:
        layer = self.num_layers - 1
        while(layer != -1):
            # it is an iterator used in case there are multiple output neurons
            it = 0
            # we iterate through every neuron in the current layer
            for neuron in self.layers[layer].neurons:
                delta = 0
                # if the current layer is NOT the ouptut layer
                if (layer != self.num_layers - 1):
                    # First, we need to update the weights of all edges stemming from this neuron
                    for a in range(0, len(neuron.weights)):
                        neuron.weights[a] = neuron.weights[a] - self.gamma * self.layers[layer+1].neurons[a].delta * neuron.value
                    # next, we need to update this delta of this neuron
                    # we iterate through all neurons in the next layer by index
                    for a in range(0,self.layers[layer+1].size):
                        # we do the summation part of the delta formula, where the value is delta_z * weight_yz
                        delta += self.layers[layer+1].neurons[a].delta * neuron.weights[a]
                    # we finally multiply by the derivative of the sigmoid function
                    delta *= neuron.value * (1 - neuron.value)
                    # we set the current neurons delta value
                    neuron.delta = delta
                else:
                    # if we're in the output layer, we simply calculate the delta by single formula
                    neuron.delta = (neuron.value - exp[it]) * neuron.value * (1 - neuron.value)
                    #print("OUTPUT DELTA: ", neuron.delta)
                    #print("OUTPUT VALUE: ", neuron.value)
                    # in this case, there are no weights to change, so we just continue
                # we increase the special iterator by 1 (for indexing purposes)
                it += 1

            layer -= 1

    # used for debugging
    def print_network(self):
        for a in range(0, self.num_layers):
            print("SIZE OF LAYER: ",self.layers[a].size)
            it = 0
            for n in self.layers[a].neurons:
                print("WEIGHTS FOR NEURON ",it," in layer ",a,": ",n.weights)
                it += 1

    def get_output(self):
        return self.layers[self.num_layers-1].get_vector()

    def print_weights(self):
        b = 0
        for layer in self.layers:
            a = 0
            print("LAYER: ",b)
            for neuron in layer.neurons:
                print("NEURON ",a," WEIGHTS = ",neuron.weights, "   DELTA = ",neuron.delta)
                a += 1
            b += 1

    def get_delta(self):
        return self.layers[self.num_layers-1].neurons[len(self.layers[self.num_layers-1].neurons)-1].delta

def random_input():
    i = 2*random.random()-.97;
    return [i - .03, i - .02, i - .01, i]


def main():

    random.seed(3223)

    data_folder = Path("C:/Users/ultim/Documents/2020/School Stuff/Research/AI/Data/")
    file_to_open = data_folder / "Timestep_1.mat"
    mat_contents = sio.loadmat(file_to_open)
    mat_U = mat_contents['U']
    mat_V = mat_contents['V']
    mat_W = mat_contents['W']
    #print("U shape: ", mat_U.shape)
    #print("V size: ", mat_V.size)
    #print("W size: ", mat_W.size)
    #print(mat_U[0, 0, 0])

    my_network = Network(input_size = 3,
                         output_size= 1,
                         num_layers= 3,
                         gamma= .1)
    n = 10000
    deltas = []
    diff = []
    x = []
    for a in range(0,n):
        inp = random_input()
        my_network.compute([inp[0], inp[1], inp[2]],[inp[3]])
        x.append(a)
        deltas.append(my_network.get_delta())
        diff.append(my_network.get_output()[0] - inp[3])
        #print("DIFFERENCE: ", diff[a])
        #print("DELTA: ", deltas[a])
        #input("Press enter to continue...")
    my_network.evalutate([0,.01,.02])
    print("OUTPUT VALUE: ", my_network.get_output())
    print("DELTA: ", my_network.get_delta())
    fig, ax = plt.subplots()
    ax.plot(x, deltas)
    ax.set(xlabel='iterations', ylabel='delta value',
           title='Plot values')

    plt.show()
    #print(my_network.get_output())
    #my_network.print_network()


if __name__ == "__main__":
    main()
