# nutil.py
# used to store the common functions among dependent scripts

# module imports
import matplotlib.pyplot as plt

# global variables
debugf = False
version = 1.0
save_path = "Out/"
nprint_default = 'NNI'
data_path = "C:/Users/ultim/Documents/2020/School Stuff/Research/AI/Data/"


# Plots a given set
def plot_set(x, y, ylabel='y', xlabel='x', title='graph'):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    plt.show()


# prints a message with a designated tag
def nprint(message, tag=nprint_default):
    print("[", tag, "] ", message)


# prints debug messages
def debug(message, data=""):
    if debugf:
        nprint(message, "DEBUG")
