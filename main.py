import numpy as np
import matplotlib.pyplot as plt

def read_pgm(path):
    with open(path, 'rb') as pgmf:
        im = plt.imread(pgmf)
    return im

if __name__ == "__main__":
    path = "/home/william/data/cybathlon/test.pgm"
    mat = read_pgm(path)
    plt.figure()
    plt.imshow(mat)
    plt.show()