# ntest.py
# used for test scripts for the functions in this program

# module imports
import matplotlib as plt
import random

# local imports
import nconsole
import nnetwork
import ndata


# returns a vector in the sequence of [a, a+.01, a+.02, a+.03] where a = [0,.97]
# used for debugging
def random_input():
    i = random.random() * .97 + .03
    return [i - .03, i - .02, i - .01, i]


# another debug function
def debug_2():
    network = nconsole.load_network("test.txt")
    if network == -1:
        print("Exiting")
    else:
        print("Woo hoo!")
    network.print_weights()


# main function used for debugging
def debug_3():
    # create a new network with the given variables
    my_network = nnetwork.Network(input_size=3,
                                  output_size=1,
                                  num_layers=2,
                                  gamma=.1)

    # n is the number of iterations
    n = 100000

    # establish some sets to be graphed for debugging purposes
    deltas = []  # list of the delta values
    diff = []  # list of the differences between calculated and accepted values
    x = []  # the x axis
    accuracy_set = []  # list of the accuracies for the network
    cost_set = []  # list of costs used for the network

    # iterate through n times, training the network over a random vector of inputs to the correct output
    for a in range(0, n):
        # generate a random vector sequence and run the compute function over it
        inp = random_input()
        my_network.compute([inp[0], inp[1], inp[2]], [inp[3]])
        # append the meta-data to the appropriate array
        x.append(a)
        deltas.append(my_network.get_delta())
        diff.append(my_network.get_output()[0] - inp[3])
        accuracy_set.append(inp)
        cost_set.append((1 / 2) * (diff[a]) ** 2)

    # test the function over an "easy to calculate" vector
    my_network.evaluate([0, .01, .02])
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
    my_network.save_network("test.txt")
    plt.show()


def test():
    data_size = 100000
    test_size = 50000
    min_accuracy = 1
    best = []
    best_file = ""
    iterator = 0
    total_tests = 5 * 4 * 3
    print("BEGINNING TESTS")
    for d in range(0, 3):
        if d == 0:
            print("U TEST")
        elif d == 1:
            print("V TEST")
        elif d == 2:
            print("W TEST")
        for i in range(2, 7):
            for num_layers in range(0, 4):
                for layer_size_override in range(0, 1):
                    iterator += 1
                    print("TEST ", iterator, " OUT OF ", total_tests)
                    name = "TEST_1_" + str(iterator) + ".txt"
                    input_set, output_set = partition(data[d], n=data_size, input_size=i)
                    test_inp, test_out = partition(data[d], n=test_size, input_size=i)
                    accuracy, rel_diff, network = run_network(i, input_set, output_set, test_inp, test_out, name=name,
                                                              num_layers=num_layers, gamma=.1)
                    print("AVERAGE % ERROR: ", accuracy * 100)
                    print("AVERAGE RELATIVE DIFFERENCE: ", rel_diff)
                    if accuracy < min_accuracy:
                        min_accuracy = accuracy
                        best = network
                        network.save_network(name)
                        best_file = name
    print("BEST FIT: " + best_file)
    print("ACCURACY: " + min_accuracy)


if __name__ == '__main__':
    debug_2()
