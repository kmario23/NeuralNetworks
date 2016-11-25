""" @author: kmario23 """

import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Input: auto-mpg.data
    """
    #prepare data
    data = np.loadtxt('auto-mpg.data', usecols=(0, 2))
    Xfull, Yfull = data[:, 0], data[:, 1]
    Xtrain, Ytrain = Xfull[0:50], Yfull[0:50]
    Xtest, Ytest = Xfull[50:100], Yfull[50:100]

    #plot data
    plt.plot(Xtrain, Ytrain)
    plt.xlabel('miles per gallon')
    plt.ylabel('displacement')
    plt.show()


if __name__ == "__main__":
    main()
