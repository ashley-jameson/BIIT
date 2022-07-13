import numpy as np
import scipy as sp
import mendeleev as mdl
import random
import collections
from max_entropy import *

ket_zero = np.array([1,0])
ket_one = np.array([0,1])

def heaviside_smooth(x,k=1):
    # logistic
    return 1. / 1. + np.exp(-2*k*x)

def rho_function(weights, dims):
    if quantum_perceptron_type == "Ising":
        y = np.multiply(np.heaviside(weights - 1) + np.heaviside(weights + 1), axis=1)
    elif quantum_perceptron_type == "Spherical":
        y = np.heaviside(np.absolute(weights) **2 - dims)
    return  y / np.sum(y)

class Perceptron():
    def __init__(self, input_vector, expectation, quantum_perceptron_type = "Spherical"):
        assert type(input_vector == np.complex64)
        assert input_vector.shape[0] == 2
        assert np.isclose(np.absolute(self.input_vector[:,0])**2, np.absolute(np.input_vector[:,1])**2,1.)
        self.input_vector = input_vector
        self.hilbert_space_dim = self.input_vector.shape[1]
        self.threshold = np.zeros(self.input_vector.shape[1])
        self.expectation = expectation
        # check axes
    def learn_weights(self):
        self.weights = MaxEntropy(self.input_vector,self.expectation)
        return

    def calculate_volume(self):
        self.perceptron_volume = np.sum(np.multiply(heaviside_smooth(self.X))*rho_function(self.weights))
        return

    def run_all(self):
        self.learn_weights
        self.X = (np.absolute(np.vdot(self.input_vector, self.weights))**2 - self.threshold) / self.hilbert_space_dim
        self.calculate_volume
        return self.perceptron_volume
