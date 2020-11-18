# nconsole.py
# This is the main .py file in the program, used as a user interface for utilizing networks

# module imports
import sys
import random
from os import path
import datetime

# local imports
import nutil
import nnetwork
import ndata


# Allows the user to either choose [Y]es or [n]o and returns the users choice
def console_confirm(initial_prompt, failure_prompt):
    choice = input(initial_prompt + " ")
    while not choice == "Y" and not choice == "n":
        choice = input(failure_prompt)
    return choice


# Generates a unique name (based on the current timestamp) for a network
def generate_name():
    dt = datetime.datetime.now()
    return "NNI_network_" + str(dt.month) + "-" + str(dt.day) + "-" + str(dt.year) + "T-" + str(dt.hour) + ":" + str(
        dt.minute)


# prints help message for running the program through the terminal (ex improper use)
def terminal_help():
    print("Usage: nconsole.py [-h|-d]")
    print("h: Help")
    print("d: Debug")


# loads a network under a given filename (must end in .txt)
def load_network(name, pth=nutil.save_path) -> nnetwork.Network:
    file_name = pth + name
    if not path.exists(file_name):
        nutil.nprint("File does not exist!")
        return None
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
        nutil.debug("NUM NEURONS: ", num_neurons)
        for b in range(0, num_neurons):
            if b == num_neurons - 1:
                network.get_layer(a).neurons.append(nnetwork.Neuron(value=1))
            else:
                network.get_layer(a).neurons.append(nnetwork.Neuron())
            if a == num_layers - 1:
                network.output_size = num_neurons
                network.get_neuron(a, b).set_weights([])
                nutil.debug("YAY")
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


# called when 'exit' command is entered
def cexit(args):
    nutil.nprint_default = 'EXT'
    if not ndata.primary_network_saved or not ndata.secondary_network_saved:
        msg = "[ WARNING ]  You still have unsaved work. Are you sure you would like to exit without saving? [Y/n]"
        ch = console_confirm(msg, "[ ERROR ]  Please enter either [Y] or [n]")
        if ch == "n":
            return 1
    nutil.nprint("Have a great day!")
    return 0


# called when 'help' command is entered
def chelp(args):
    nutil.nprint_default = 'HLP'
    if len(args) == 1:
        nutil.nprint('LIST OF COMMANDS AND THEIR USAGE:')
        nutil.nprint('help [cmd]')
        nutil.nprint('exit')
        nutil.nprint('load [-s] [-p path] -n name')
        nutil.nprint('save [-s] [-p path] [-n name]')
        nutil.nprint('train [-n] [-a [-p path] [-n name]] [-c]')
        nutil.nprint('test [-i iterations]')
        nutil.nprint('config')
        nutil.nprint("For more detailed usages, use 'help [cmd]'")
        nutil.nprint("Please note, the command name MUST be first, the order of the optional arguments can be in any ")
        nutil.nprint("order, however if they need to be proceeded with a variable, then they must be.")
        return
    elif len(args) > 2:
        nutil.nprint("Usage: 'help [cmd]")
        return
    if args[1] == 'exit':
        nutil.nprint("Exits the program. You will be prompted if there is any unsaved files open.")
    elif args[1] == 'load':
        nutil.nprint("Load a network from a file:")
        nutil.nprint("[-s] is an optional input that indicates to load the secondary network. If this is not present,")
        nutil.nprint("the primary network will be loaded instead.")
        nutil.nprint("[-p path] is an optional input that designates what path to load the file from. By default, path")
        nutil.nprint("is set to Out/, and unless an absolute path, the program will look locally for path.")
        nutil.nprint("-n name designates the name of the file to load. This cannot be empty, or else an error message")
        nutil.nprint("will be printed.")
    elif args[1] == 'save':
        nutil.nprint("Saves the current network to a file:")
        nutil.nprint("[-s] is an optional input that indicates to save the secondary network. If this is not present,")
        nutil.nprint("the primary network will be saved instead.")
        nutil.nprint("[-p path] is an optional input that designates what path to save the file to. By default, path")
        nutil.nprint("is set to Out/, and unless an absolute path, the program will look locally for path.")
        nutil.nprint("[-n name] is an optional input that designates the name of the file. The file will be saved as ")
        nutil.nprint("<name>.txt, however if it's left empty, it will be a procedurally generated name.")
    elif args[1] == 'train':
        nutil.nprint("Trains a neural network over sampled data:")
        nutil.nprint("[-n] is an optional input that when present, indicates that a new network will be created.")
        nutil.nprint("[-a] is an optional input that when present, indicates that the generated network will be ")
        nutil.nprint("automatically saved. This can be additionally followed by [-p path] and/or [-n name]. See ")
        nutil.nprint("'help save' for usage of the -p and -n options.")
        nutil.nprint("[-c] is an optional input that will execute config mode. See 'help config' for more information.")
    elif args[1] == 'test':
        nutil.nprint("Tests a neural network over test data:")
        nutil.nprint("[-i iterations] is an optional input that indicates the designated number of iterations to test")
        nutil.nprint("over. By default, the iterations is set to 100,000.")
    elif args[1] == 'config':
        nutil.nprint("Executes 'config mode', which allows the user to change configuration settings with the program.")
    elif args[1] == 'help':
        nutil.nprint('Listen here wiseguy...')
    else:
        nutil.nprint("Command not recognized. Type 'help' for a list of available commands.")


# called when 'load' command is entered
def cload(args):
    nutil.nprint_default = 'LOD'
    load_name = ""
    load_secondary = False
    load_path = nutil.save_path
    size = len(args)
    for index in range(0, size):
        if args[index] == '-s':
            load_secondary = True
        if args[index] == '-p' and index < size - 1:
            index += 1
            load_path = args[index]
        if args[index] == '-n' and index < size - 1:
            index += 1
            load_name = args[index]
    if load_name == "":
        nutil.nprint("Load name was empty. Please ensure you use 'load -n name [-p path] [-s]' to load.")
        return -1
    if load_secondary:
        ndata.secondary_network = load_network(load_name, load_path)
        if ndata.secondary_network is None:
            nutil.nprint("Unable to load network. Please check file arguments.")
            return
        ndata.secondary_file_name = load_path + load_name
        ndata.secondary_network_saved = True
    else:
        primary_network = load_network(load_name, load_path)
        if primary_network is None:
            nutil.nprint("Unable to load network. Please check file arguments.")
            return
        ndata.primary_file_name = load_path + load_name
        ndata.primary_network_saved = True
    return


# called when 'save' command is entered
def csave(args):
    nutil.nprint_default = 'SAV'
    save_secondary = False
    save_path = nutil.save_path
    save_name = generate_name()
    size = len(args)
    for index in range(0, size):
        if args[index] == '-s':
            save_secondary = True
        if args[index] == '-p' and index < size - 1:
            index += 1
            save_path = args[index]
        if args[index] == '-n' and index < size - 1:
            index += 1
            save_name = args[index]
    if save_secondary:
        if path.exists(save_path + save_name):
            overwrite = console_confirm("[ SAV ]  Filename exists already, would you like to overwrite? [Y/n]",
                                        "[ SAV ]  Please select either [Y] or [n]")
            if overwrite == 'n':
                save_name += "(2)"
        ndata.secondary_file_name = save_name + save_path
        ndata.secondary_network_saved = True
        save_network(ndata.secondary_network, save_name, save_path)
    else:
        if path.exists(save_path + save_name):
            overwrite = console_confirm("[ SAV ]  Filename exists already, would you like to overwrite? [Y/n]",
                                        "[ SAV ]  Please select either [Y] or [n]")
            if overwrite == 'n':
                save_name += "(2)"
        ndata.primary_file_name = save_name + save_path
        ndata.primary_network_saved = True
        save_network(ndata.primary_network, save_name, save_path)
    msg = "File was successfully saved at: " + save_path + save_name
    nutil.nprint(msg)
    # called when 'config' command is entered


def cconfig(args):
    nutil.nprint_default = 'CFG'
    nutil.nprint('CONFIG')


# called when 'train' command is entered
def ctrain(args):
    nutil.nprint_default = 'TRN'
    nutil.nprint('TRAIN')


# called when 'test' command is entered
def ctest(args):
    nutil.nprint_default = 'TST'
    nutil.nprint('TEST')


# main function
def console(args):
    running = True
    if len(args) == 2:
        if args[1] == '-h':
            terminal_help()
        if args[1] == '-d':
            nutil.debug = True
        if len(args) > 2:
            terminal_help()
            return -1
    print("-===NEURAL NETWORK INTERFACE ", nutil.version, "===-\n")
    nutil.nprint('startup procedure initialized.')
    # TODO
    # Preprocessing
    command = ""
    while running:
        print("------------------------")
        try:
            command = input("[ NNI ]  ")
        except KeyboardInterrupt:
            nutil.nprint('Keyboard Interrupt. Exiting')
            cexit()
            return 1
        print()
        args = command.split()
        nutil.nprint_default = 'NNI'
        if args[0] == 'save':
            csave(args)
        elif args[0] == 'load':
            cload(args)
        elif args[0] == 'exit':
            exit_choice = cexit(args)
            if exit_choice == 0:
                return 0
        elif args[0] == 'help':
            chelp(args)
        elif args[0] == 'config':
            cconfig(args)
        elif args[0] == 'test':
            ctest(args)
        elif args[0] == 'train':
            ctrain(args)
        else:
            nutil.nprint("Command not recognized. Type 'help' to see available commands and usages.")


if __name__ == "__main__":
    random.seed(3223)
    return_code = console(sys.argv)
    print("Program exited with return code", return_code)
