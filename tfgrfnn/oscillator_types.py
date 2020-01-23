import numpy as np
import tensorflow as tf

class hopfpar():

    def __init__(self, alpha=0.0, beta1=-1.0, beta2=0.0, 
                delta1=0.0, delta2=0.0, epsilon = 0.0):
    
        self.alpha = beta2
        self.beta1 = beta1
        self.delta1 = delta1
        self.delta2 = delta2
        self.epsilon = epsilon
