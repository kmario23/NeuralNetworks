""" @author: kmario23 """

import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Input: auto-mpg.data
    """
    # prepare data
    data = np.loadtxt('auto-mpg.data', usecols=(0, 2))
    Xfull, Yfull = data[:, 0], data[:, 1]
    Xtrain, Ytrain = Xfull[0:50], Yfull[0:50]
    Xtest, Ytest = Xfull[50:100], Yfull[50:100]

    # plot data
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(5, 30, 1))
    ax.set_yticks(np.arange(50, 500, 40))
    plt.scatter(Xtrain, Ytrain)
    plt.xlabel('miles per gallon')
    plt.ylabel('displacement')
    plt.grid(True)
    plt.show()

    # observing the plot suggests that we need at least
    # quadratic polynomial to appropriately fit the data

    # fit using first order polynomial
    # (w * (np.transpose(Xtrain) * Xtrain)) = (np.transpose(Xtrain) * Ytrain)


if __name__ == "__main__":
    main()
