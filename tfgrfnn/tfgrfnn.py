import numpy as np
import tensorflow as tf

from oscillator_types import hopf
from connection_learning_rule_types import phase_lock 

class oscillators():

    def __init__(self, name = '', 
                    osctype = hopf(),
                    nosc = 256,
                    freqlims = (0.12, 8.0),
                    freqspace = 'log',
                    initconds = np.zeros((256, )),
                    save_steps = True):

        self.name = name
        self.osctype = osctype
        self.params = self.osctype.params()
        self.freqspace = freqspace
        self.freqlims = freqlims
        self.nosc = nosc
        if self.freqspac == 'log':
            self.freqs = np.logspace(np.log10(self.freqlims[0]),
                            np.log10(self.freqlims[1]), 
                            self.nosc)
            if self.nosc > 1:
                self.freqdelta = self.freqs[1] / self.freqs[0]
            else:
                self.freqdelta = 1
        elif self.freqspac == 'lin':
            self.freqs = np.linpace(self.freqlims[0],
                            self.freqlims[1],
                            self.nosc)
            if self.nosc > 1:
                self.freqdelta = self.freqs[1] - self.freqs[0]
            else:
                self.freqdelta = 1
        self.initconds = initconds
        self.save_steps = save_steps
        self.connections = []


class connection():

    def __init__(self, name = '', 
                    source = None,
                    target = None,
                    conn_type = None,
                    amplitude = 1.0,
                    range_ = 0.12,
                    learn = True,
                    complex_ = True,
                    save_steps = True):

        self.name = name
        self.source = source
        self.target = target
        self.conn_type = conn_type 
        self.save_steps = save_steps

def connect(source = None, target = None, conn_matrix = None, conn_type = None,
        learn = True, ): 
    target.connections.append(conn_matrix) 

    return target
        
