# scipy is the libary used to read from a MATLAB file
import scipy.io as sio
# pathlib is a standard library from Python 3.4 that allows for proper path generation
from pathlib import Path












def main():
    data_folder = Path("C:/Users/ultim/Documents/2020/School Stuff/Research/AI/Data/")
    file_to_open = data_folder / "Timestep_1.mat"
    mat_contents = sio.loadmat(file_to_open)
    mat_U = mat_contents['U']
    mat_V = mat_contents['V']
    mat_W = mat_contents['W']
    print("U shape: ", mat_U.shape)
    print("V size: ", mat_V.size)
    print("W size: ", mat_W.size)

    print(mat_U[0,0,0])

if __name__ == "__main__":
    main()