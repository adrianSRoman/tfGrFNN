import numpy as np
import tensorflow as tf

from oscillator_types import hopf
from connection_learning_rule_types import phase_lock 
from ode_functions import xdot_ydot, crdot_cidot

class oscillators():

    def __init__(self, name = '', 
                    osctype = hopf(),
                    nosc = 256,
                    freqlims = (0.12, 8.0),
                    freqspacing = 'log',
                    initconds = np.zeros((256, )),
                    save_steps = True):

        self.name = name
        self.osctype = osctype
        self.params = self.osctype.params()
        self.freqspacing = freqspacing
        self.freqlims = freqlims
        self.nosc = nosc
        if self.freqspacing == 'log':
            self.freqs = np.logspace(np.log10(self.freqlims[0]),
                            np.log10(self.freqlims[1]), 
                            self.nosc)
            if self.nosc > 1:
                self.freqdelta = self.freqs[1] / self.freqs[0]
            else:
                self.freqdelta = 1
        elif self.freqspacing == 'lin':
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
       

class Model():

    def __init__(self, name = '',
                    layers = None,
                    zfun = xdot_ydot,
                    cfun = crdot_cidot,
                    time = tf.range(1, delta = 0.01, dtype = tf.float64),
                    dt = 0.01):

        self.layers = layers
        self.time = time
        self.zfun = zfun
        self.cfun = cfun
        self.dt = dt

    def integrate(self):

        all_layers_initconds, all_layers_connections = [(layer.initconds, layer.connections) for layer in self.layers]
        all_conns_initconds = [[connection.initconds for connection in layer_connections] for layer_connections in all_layers_connections]

        def odeRK4(all_init_conds, time):

            time_plus_half_dt = tf.add(time, self.dt/2)

            all_layers_state, all_conns_state = all_init_conds

            all_layers_k1 = [self.zfun(time, layer_state, M.layers[ilayer].params) for ilayer, layer_state in enumerate(all_layers_state)]
            all_conns_k1 = [[self.cfun(time, conn_state, M.layers[ilayer].connections[iconn].params) 
                                for iconn, conn_state in enumerate(all_conns_state[ilayer])] 
                            for ilayer in range(len(all_layers_state))]
            all_layers_k2 = [self.zfun(time_plus_half_dt, tf.add(layer_state, tf.scalar_mul(self.dt, layer_k1/2)), M.layers[ilayer].params) 
                            for ilayer, (layer_state, layer_k1) in enumerate(zip(all_layers_state, all_layers_k1))]
            all_conns_k2 = [[self.cfun(time_plus_half_dt, tf.add(conn_state, tf.scalar_mul(self.dt, conn_k1/2)), M.layers[ilayer].connections[iconn].params) 
                                for iconn, (conn_state, conn_k1) in enumerate(zip(all_conns_state[ilayer], all_conns_k1[ilayer]))] 
                            for ilayer in range(len(all_layers_state))]
            all_layers_k3 = [self.zfun(time_plus_half_dt, tf.add(layer_state, tf.scalar_mul(self.dt, layer_k2/2)), M.layers[ilayer].params) 
                            for ilayer, (layer_state, layer_k2) in enumerate(zip(all_layers_state, all_layers_k2))]
            all_conns_k3 = [[self.cfun(time_plus_half_dt, tf.add(conn_state, tf.scalar_mul(self.dt, conn_k2/2)), M.layers[ilayer].connections[iconn].params) 
                                for iconn, (conn_state, conn_k2) in enumerate(zip(all_conns_state[ilayer], all_conns_k2[ilayer]))] 
                            for ilayer in range(len(all_layers_state))]
            all_layers_k4 = [self.zfun(tf.add(time, self.dt), tf.add(layer_state, tf.scalar_mul(self.dt, layer_k3)), M.layers[ilayer].params) 
                            for ilayer, (layer_state, layer_k3) in enumerate(zip(all_layers_state, all_layers_k3))]
            all_conns_k4 = [[self.cfun(tf.add(time, self.dt), tf.add(conn_state, tf.scalar_mul(self.dt, conn_k3)), M.layers[ilayer].connections[iconn].params) 
                                for iconn, (conn_state, conn_k3) in enumerate(zip(all_conns_state[ilayer], all_conns_k3[ilayer]))] 
                            for ilayer in range(len(all_layers_state))]
            all_layers_state = [tf.add(layer_state, tf.divide(tf.add_n([layer_k1, tf.scalar_mul(2, layer_k2), tf.scalar_mul(2, layer_k3), layer_k4]), 6))
                                for layer_state, layer_k1, layer_k2, layer_k3, layer_k4 
                                    in zip(all_layers_state, all_layers_k1, all_layers_k2, all_layers_k3, all_layers_k4)] 
            all_conns_state = [[tf.add(conn_state, tf.divide(tf.add_n([conn_k1, tf.scalar_mul(2, conn_k2), tf.scalar_mul(2, conn_k3), conn_k4]), 6))
                                    for conn_state, conn_k1, conn_k2, conn_k3, conn_k4 
                                        in zip(all_conns_state[ilayer], all_conns_k1[ilayer], all_conns_k2[ilayer], all_conns_k3[ilayer], all_conns_k4[ilayer])]
                                for ilayer in range(len(all_layers_state))]

            return all_layers_state, all_conns_state

        all_layers_steps, all_conns_steps = tf.scan(odeRK4, time, (all_layers_initconds, all_conns_initconds))
        
        for ilayer in range(len(self.layers)):
            self.layers[ilayer].saved_states = all_layers_steps[ilayer]
            for iconn in range(len(self.layers[ilayer].connections)):
                self.layers[ilayer].connections[iconn].saved_states = all_conns_steps[ilayer][iconn]

