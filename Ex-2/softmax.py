""" @author: kmario23
Fast & stable implementation of Softmax function to ensure numerical stability.
"""
import numpy as np

def stable_softmax(x):
    """
    Input : some vector 'x'
    Output: a vector with entries that sum to 1;

    Intuitively, it converts a vector of real entries to a vector of probabilities, that sum to 1

    To avoid overflow & underflow while taking exponents, use the following trick.
    transform the vector x to z, where:
        z = x - max(x)

    Why this works?
        Simple algebra shows that value of softmax is not changed if we add/subtract 
        a scalar value from the input vector.

        once we do: x - max(x), the largest element in the vector becomes 0; exp(0) would return 1
        This solves the problem of overflow.

        Also, at least one term would be 1 in the denominator. So, there won't be any issue of dividing by zero.

        Still, underflow can occur in the numerator. And the solution is to use: log(stable_softmax(x))
    """

    z = x - np.max(x)
    exp_z = np.exp(z)
    distrib = exp_z / sum(exp_z)

    return distrib


if __name__ == '__main__':
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    w = np.array(v)
    dist = stable_softmax(w)
    print(dist)

