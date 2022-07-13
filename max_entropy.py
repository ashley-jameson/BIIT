#!/usr/bin/python3
# @Author: Ashika Jayanthy

import numpy as np

class MaxEntropy():
    def __init__(self, observed, expected):
        self.observed = observed
        self.expected = expected
        self.weights = np.ones(self.input.shape[1])

    def calculate_weights():
        #check axes
        cost = 1
        while cost >= 1e-10:
            P_expected = np.exp(self.expected) / np.sum(np.exp(self.expected),axis=1)
            P_observed = np.exp(self.observed * self.weights) / np.sum(np.exp(self.observed * self.weights), axis=1)
            cost = np.log(P_expected / P_observed, axis=1)
            self.weights += cost
        return self.weights
