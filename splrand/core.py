"""
This file contains the actual code for the package splrand.
It aims to generate univariate random numbers according to a given
distribution using splines.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class ProbabilityDensityFunction:
    """
    Class to extract random values from a pdf and/or evaluate
    the pdf for given intervals/data points
    """


    def __init__(self, x, y, k = 3):
        """
        defines a normalised pdf self.spline from the input data and
        computes the corresponding cumulative distribution self.cdf

        Arguments:
        -----------

        x: np.array
            independent variable

        y: np.array
            values of the pdf on x

        k: int
            degree of the polynomial used for the spline interpolation;
            defaults to 3
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be of type np.ndarray")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be of type np.ndarray")
        if not int(x.ndim) == 1:
            raise TypeError("x must be 1-dimensional")
        if not int(y.ndim) == 1:
            raise TypeError("y must be 1-dimensional")
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length, but have lengths {len(x)} and {len(y)}")



        self.x = x
        self.y = y

        self.x_min = min(x)
        self.x_max = max(x)
        self.y_min = min(y)         #should be 0
        self.y_max = max(y)

        self.k = k


        self.spline = InterpolatedUnivariateSpline(self.x, self.y, k = self.k)

        norma = self.spline.integral(self.x_min, self.x_max)
        self.y /= norma

        self.spline = InterpolatedUnivariateSpline(self.x, self.y, k = self.k)

        self.cdf = np.array([self.spline.integral(self.x_min, t) for t in self.x])

    def __str__(self):
        return f"{self.__class__.__name__}"

    def evaluate(self, x):
        """
        evaluate the pdf for single value or array x

        Arguments:
        ----------

        x: float, int or np.array
            x-value(s) for which to evaluate the pdf

        returns: float or np.array
            evaluated pdf-value(s)
        """
        return self.spline(x)


    def sample(self, n):
        """
        sample n x-values from the pdf; the spacing is equal to the one
        of input x

        Arguments:
        ----------

        n: int
            number of desired sample-values

        returns: float or np.array
            n samples drawn from the pdf
        """

        samples = np.zeros(n)
        uniform_samples = np.random.uniform(low = 0, high = 1, size = n)

        indices = np.searchsorted(self.cdf, uniform_samples)

        for i, idx in enumerate(indices):
            if idx > 0 and (idx == len(self.cdf) or np.abs(self.cdf[idx-1] - uniform_samples[i]) < np.abs(self.cdf[idx] - uniform_samples[i])):
                samples[i] = self.x[idx-1]
            else:
                samples[i] = self.x[idx]

        return samples



if __name__ == "__main__":
    from matplotlib import pyplot as plt

    x = np.linspace(-5, 5, 200)
    y = np.zeros(200)

    for i, v in enumerate(x):
        if v < 0:
            y[i] = 0.4*x[i]+2
        else:
            y[i] = -0.4*x[i] +2

    distr = ProbabilityDensityFunction(x, y)

    print(distr.evaluate(0))
    print(distr.sample(7))

    samples = distr.sample(20000)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.hist(samples, bins = 25)
    plt.show()
