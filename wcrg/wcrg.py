import numpy as np
import pywt

# for d = 2 only

class EstimateEnergy():
    def __init__(self,phi_0, num_j, a=1):
        self.phi_0 = phi_0
        self.phi_array = np.zeros((num_j,2))
        self.a = a

    def wavelet_field(self):
        self.phi_0_bar =
