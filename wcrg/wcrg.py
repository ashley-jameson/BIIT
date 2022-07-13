import numpy as np
import pywavelets

# for d = 2 only

class EstimateEnergy():
    def __init__(self,phi_0, num_j, a=1):
        self.phi_0 = phi_0
        self.j = np.arange(0,num_j)

    def wavelet_field(self):
        self.phi_0_bar = 
