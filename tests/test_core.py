"""
Unit tests for the ProbabilityDensityFunction class
"""

from splrand.core import ProbabilityDensityFunction
import numpy as np
from scipy.integrate import quad


def test_gaussian():
    """
    Defining a Gaussian pdf and checking its properties
    """

    x = np.linspace(-5, 5, 201)
    y = np.exp(-1/2*x**2)

    Gaussian = ProbabilityDensityFunction(x, y)

    cum_sum = 0
    for k, xk in enumerate(x):
        yk = Gaussian.evaluate(xk)
        assert yk >= 0, "Negative value of pdf encountered"
        if k > 0:
            cum_sum += yk*(xk - x[k-1])

    assert np.abs(cum_sum - 1) < 1e-6, "Normalisation fails"


def test_triangular():
    """
    Defining a triangular pdf and checking its properties
    """

    x = np.linspace(-5, 5, 200)
    y = np.zeros(200)

    for i, v in enumerate(x):
        if v < 0:
            y[i] = 0.4*x[i]+2
        else:
            y[i] = -0.4*x[i] +2

    triangular = ProbabilityDensityFunction(x, y)

    cum_sum = 0
    for k, xk in enumerate(x):
        yk = triangular.evaluate(xk)
        assert yk >= 0, "Negative value of pdf encountered"
        if k > 0:
            cum_sum += yk*(xk - x[k-1])

    assert np.abs(cum_sum - 1) < 1e-3, "Normalisation fails"

test_gaussian()
test_triangular()
