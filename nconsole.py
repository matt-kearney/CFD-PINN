# nconsole.py
# This is the main .py file in the program, used as a user interface for utilizing networks

# module imports
import sys
import random
from os import path

# local imports
import nutil
import nnetwork


# prints help message for running the program through the terminal (ex improper use)
def terminal_help():
    print("Usage: nconsole.py [-h|-d]")
    print("h: Help")
    print("d: Debug")


# loads a network under a given filename (must end in .txt)
def load_network(name, pth=nutil.save_path):
    file_name = pth + name
    if not path.exists(file_name):
        print("File does not exist!")
        return -1
    network = nnetwork.Network()
    f = open(file_name, "r")
    buffer = f.read()
    buffer_list = buffer.split()
    gamma = float(buffer_list[0])
    num_layers = int(buffer_list[1])
    network.num_layers = num_layers
    network.gamma = gamma
    network.layers = []
    network.input_size = buffer_list[3]
    i = 2  # index of the list[]
    for a in range(0, num_layers):
        num_neurons = int(buffer_list[i])
        i += 1
        if a == num_layers - 1:
            network.layers.append(nnetwork.Layer(size=num_neurons, layer_num=a, load=True))
        else:
            network.layers.append(nnetwork.Layer(size=num_neurons - 1, layer_num=a, load=True))
        print("NUM NEURONS: ", num_neurons)
        for b in range(0, num_neurons):
            if b == num_neurons - 1:
                network.get_layer(a).neurons.append(nnetwork.Neuron(value=1))
            else:
                network.get_layer(a).neurons.append(nnetwork.Neuron())
            if a == num_layers - 1:
                network.output_size = num_neurons
                network.get_neuron(a, b).set_weights([])
                print("YAY")
                break
            elif b > 0:
                network.layer_size = num_neurons
            num_weights = int(buffer_list[i])
            i += 1
            weights = []
            for w in range(0, num_weights):
                weights.append(float(buffer_list[i]))
                i += 1
            network.get_neuron(a, b).set_weights(weights)
    return network


# saves the current network and all values to file name (name needs to end with .txt)
# it follows the format:
# GAMMA
# NUM_LAYERS \n
# NUM_NEURONS \n
# WEIGHTS OF NEURON 1
# WEIGHTS OF NEURON 2
# ...
# <REPEAT FOR ALL NEURONS IN LAYER>
# <REPEAT FOR ALL LAYERS IN NETWORK>
def save_network(network, name, pth=nutil.save_path):
    # update filename
    file_name = pth + name
    # create a new file of the given name
    f = open(file_name, "w")

    f.write(str(network.gamma) + " ")
    f.write(str(network.num_layers) + " ")
    # iterate through the network options and save the info
    current_layer = 0
    for layer in network.layers:
        if current_layer == len(network.layers) - 1:
            f.write(str(layer.size) + " ")
        else:
            f.write(str(layer.size + 1) + " ")
        current_layer += 1
        for neuron in layer.neurons:
            f.write(str(len(neuron.weights)) + " ")
            for weight in neuron.weights:
                f.write(str(weight) + " ")
    f.write("E")
    f.close()
    nutil.debug("Network successfully saved to ", name)


# main function
def console(args):
    # debug()
    # print("Hello world!")
    running = True
    if args[1] == '-h':
        terminal_help()
    if args[1] == '-d':
        debug = True
    if len(args) > 1:
        terminal_help()
        return -1
    print("-===NEURAL NETWORK INTERFACE ", nutil.version, "===-\n")
    print("[STARTUP]")
    print("Type 'help' for a list of commands")
    while running:
        command = input("[NNI] ")
        running = False


if __name__ == "__main__":
    random.seed(3223)
    return_code = console(sys.argv)
    print("Program exited with return code ", return_code)
