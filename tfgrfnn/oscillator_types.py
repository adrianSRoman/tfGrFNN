import numpy as np
import tensorflow as tf

class canonical_hopf():

    def __init__(self, alpha=0.0, beta1=-1.0, beta2=0.0, epsilon = 0.0):
    
        self.alpha = tf.constant(alpha, dtype=tf.float64)
        self.beta1 = tf.constant(beta1, dtype=tf.float64)
        self.beta2 = tf.constant(beta2, dtype=tf.float64)
        self.epsilon = tf.constant(epsilon, dtype=tf.float64)

    def params(self):
        params_dict = {'alpha': self.alpha, 
                        'beta1': self.beta1, 
                        'beta2': self.beta2, 
                        'epsilon': self.epsilon}
        return params_dict

    def __repr__(self):

        return "<canonical hopf>"
