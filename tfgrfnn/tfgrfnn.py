import numpy as np
import tensorflow as tf

from oscillator_types import hopf_params

class oscillator_layer():

    def __init__(self, name='', 
                    osctype='Hopf',
                    freqlims=(0.12, 8.0),
                    freqspac='log',
                    nosc=256,
                    initconds=np.zeros((256,)),
                    save_states=True):

        self.name = name
        self.osctype = osctype
        if self.osctype == 'Hopf'
            self.params = hopf_params()
        self.freqspac = freqspac
        self.freqlims = freqlims
        self.nosc = nosc
        if self.freqspac == 'log':
            self.freqs = np.logspace(np.log10(self.freqlims[0]),
                    np.log10(self.freqlims[1]),self.nosc)
            if self.nosc > 1:
                self.freqdelta = self.freqs[1]/self.freqs[0]
            else:
                self.freqdelta = 1
        elif self.freqspac == 'lin':
            self.freqs = np.linpace(self.freqlims[0],
                    self.freqlims[1],self.nosc)
            if self.nosc > 1:
                self.freqdelta = self.freqs[1]-self.freqs[0]
            else:
                self.freqdelta = 1
        self.initconds = initconds
        self.save_states = save_states
        self.connections = []


class connection_matrix():

    def __init__(self, ):
