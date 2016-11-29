""" @authors: kmario23 """

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import animation


def load_data():
    """
    Input: n instances of 2-D points (x,y)
    """
    #a) prepare data
    data = np.loadtxt('data_kmeans.txt', usecols=(0, 1))
    Xn, Yn = data[:, 0], data[:, 1]
    #print(len(Xn), len(Yn))

    # plot data
    fig = plt.figure()
    ax = fig.gca()
    plt.scatter(Xn, Yn)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2-D plot of data')
    plt.grid(True)
    plt.show()

    return data


def initialize_centroids(k):
    """randomly initialize 'k' cluster centers"""
    return np.random.randn(k, 2)


def determine_closest_centroid(data, centroids):
    """determines the index to the nearest centroid for each point
	Output: 'index' array
	"""
    l2_distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
	# get the index to closeset centroid
    return np.argmin(l2_distances, axis=0)

def move_centroid(data, closest_index, k_centroids):
	"""
	gives new centroids
	"""
	return np.array([data[closest_index==k].mean(axis=0) for k in range(k_centroids.shape[0])])


def plot_final_cluster(data, final_centroids):
    # plot data & clusters
    fig = plt.figure()
    ax = fig.gca()
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2-D plot of data')
    plt.grid(True)
    plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='r', s=100)
    plt.show()


#Exercise 4.1
def main():
    data = load_data()
    k = 2
    k_centroids = initialize_centroids(k)
    converged = False
    prev_centroids = k_centroids
    runs = 0

    while not converged:
        runs += 1
        closest_index = determine_closest_centroid(data, prev_centroids)
        new_centroids = move_centroid(data, closest_index, prev_centroids)
        #print(runs, ": prev centroid: ", prev_centroids)
        #print(runs, ": new centroid: ", new_centroids)

        if((prev_centroids - new_centroids).sum() == 0):
            plot_final_cluster(data, new_centroids)
            converged = True
            print("Done")
        else:
            print("iteration: ", runs)
            prev_centroids =  new_centroids


if __name__ == "__main__":
    main()

