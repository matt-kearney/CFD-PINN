# ndata.py
# data preprocessing and processing functions

# module imports
import numpy as np
import scipy.io as sio
import random

from pathlib import Path

# local imports
import nutil
import nnetwork

# instance variables for the script
primary_network = None
primary_network_saved = True
primary_file_name = ""

secondary_network = None
secondary_network_saved = True
secondary_file_name = ""

dataset = []


# normalizes n on the scale of [-1,1] based on min/max [min,max]
def normalize(dmin, dmax, n):
    drange = dmax - dmin
    n = n - dmin
    return n / drange


# partitions a set of data (in [t][x][y][z] format) into input and output arrays with given sizings
# returns an input set, output set, and input/output sets for testing
def partition(n, input_size):
    nutil.debug("Partitioning")
    input_set = []
    output_set = []
    for a in range(0, n):
        t = int(random.random() * (10 - input_size - 1))
        temp = []
        x = int(random.random() * 120)
        y = int(random.random() * 20)
        z = int(random.random() * 2200)
        for b in range(t, input_size + t):
            temp.append(dataset[b][x][y][z])
        input_set.append(temp)
        temp = [dataset[t + 1 + input_size][x][y][z]]
        output_set.append(temp)
    return input_set, output_set


# preprocessing of the script
# returns a matrix of ALL the data
# right now it's just the U part
def preprocessing():
    nutil.debug("preprocessing")
    random.seed(3223)

    data_folder = Path(nutil.data_path)

    # file_to_open = data_folder / "Timestep_1.mat"
    mat_u_all = []
    mat_v_all = []
    mat_w_all = []
    for a in range(1, 11):
        file = "Timestep_" + str(a) + ".mat"
        file_to_open = data_folder / file
        mat_contents = sio.loadmat(file_to_open)
        mat_u_all.append(mat_contents['U'])
        mat_v_all.append(mat_contents['V'])
        mat_w_all.append(mat_contents['W'])
    data = [mat_u_all, mat_v_all, mat_w_all]
    dmax = -np.inf
    dmin = np.inf
    for vec in data:
        for t in vec:
            for x in t:
                for y in x:
                    for z in y:
                        if z > dmax:
                            dmax = z
                        if z < dmin:
                            dmin = z
    nutil.debug("NORMALIZING")
    size = 10 * 120 * 60 * 2200
    iterator = 0
    for v in range(0, len(data)):
        for t in range(0, len(data[v])):
            print(iterator / size * 100, "%")
            for x in range(0, len(data[v][t])):
                for y in range(0, len(data[v][t][x])):
                    for z in range(0, len(data[v][t][x][y])):
                        iterator += 1
                        data[v][t][x][y][z] = normalize(dmin, dmax, data[v][t][x][y][z])
    return data


# creates a new neural network given the parameters for a new network (for the constructor of the Network Object)
# along with the
# TO BE REMOVED
def run_network(input_size, input_set, output_set, test_inp, test_out, output_size=1, name="default.txt", num_layers=0,
                gamma=.1, layer_size_override=0, iterations=1):
    my_network = nnetwork.Network(input_size=input_size,
                                  output_size=output_size,
                                  num_layers=num_layers,
                                  gamma=gamma,
                                  layer_size_override=layer_size_override)
    if input_size != len(input_set[0]):
        print("SIZE DISCREPANCY!")
        return
    print("RUNNING NETWORK: " + name)

    diff = []
    cost_set = []
    x = []
    deltas = []
    for it in range(0, iterations):
        for a in range(0, len(input_set)):
            x.append(a + len(input_set) * it)
            my_network.compute(input_set[a], output_set[a])
            diff.append(my_network.get_output()[0] - output_set[a])

            cost_set.append((1 / 2) * (diff[a + len(input_set) * it]) ** 2)
            temp = []
            for b in range(0, input_size):
                temp.append(input_set[a][b])
            temp.append(output_set[a])
            deltas.append(my_network.get_delta())
    accuracy = my_network.accuracy(test_inp, test_out)
    # plot_set(x, cost_set, xlabel='Iterations', ylabel='Cost Function', title='Cost Function Over x Iterations')
    return accuracy[0], accuracy[1], my_network
