import numpy as np
import tensorflow as tf

from oscillator_types import canonical_hopf
from connection_learning_rule_types import phase_lock 
from ode_functions import xdot_ydot, crdot_cidot

class oscillators():

    def __init__(self, name = '', 
                    osctype = canonical_hopf(),
                    nosc = 256,
                    freqlims = (0.12, 8.0),
                    freqspacing = 'log',
                    initconds = None,
                    savesteps = True):

        self.name = name
        self.osctype = osctype
        self.freqspacing = freqspacing
        self.freqlims = freqlims
        self.nosc = nosc
        if self.freqspacing == 'log':
            self.freqs = np.logspace(np.log10(self.freqlims[0]),
                            np.log10(self.freqlims[1]), 
                            self.nosc)
        elif self.freqspacing == 'lin':
            self.freqs = np.linpace(self.freqlims[0],
                            self.freqlims[1],
                            self.nosc)
        self.params = self.osctype.params()
        self.params['freqs'] = tf.constant(self.freqs, dtype=tf.float64)
        self.params['ones'] = tf.constant(1, dtype=tf.float64, shape=(self.nosc,))
        if initconds == None:
            self.initconds = tf.constant(0.9, dtype=tf.complex128, shape=(self.nosc,))
        else:
            self.initconds = tf.constant(initconds)
        self.savesteps = savesteps
        self.connections = [connection(typestr='null', source=self, target=self)]

    def __repr__(self):
        return "<Layer with %s %s oscillators and %s connections>" % (self.nosc, self.osctype, len(self.connections))

class connection():

    def __init__(self, name = '', 
                    source = None,
                    target = None,
                    typestr = None,
                    learn = False,
                    complex_ = True,
                    savesteps = True):

        self.name = name
        self.source = source
        self.target = target
        self.typestr = typestr
        if self.typestr == 'null' :
            self.typeint = tf.constant(0)
            self.matrix = tf.constant(0, dtype=tf.complex128, shape=(self.target.nosc,self.source.nosc))
        self.savesteps = savesteps

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
                    dt = tf.constant(0.01, dtype=tf.float64)):

        self.layers = layers
        self.time = time
        self.zfun = zfun
        self.cfun = cfun
        self.dt = dt
        self.half_dt = dt/2
        if self.zfun == xdot_ydot:
            for layer in self.layers:
                layer.params['ones'] = tf.concat([layer.params['ones'], layer.params['ones']], axis=0)
                layer.params['freqs'] = tf.concat([layer.params['freqs'], layer.params['freqs']], axis=0)
                layer.initconds = tf.concat([tf.math.real(layer.initconds), tf.math.imag(layer.initconds)], axis=0)
                layer.nosc = layer.nosc*2
                for conn in layer.connections:
                    conn.matrix = tf.concat([tf.math.real(conn.matrix), tf.math.imag(conn.matrix)], axis=0)
        #for layer in self.layers:
        #    if layer.savesteps:
        #        layer.allsteps = []
        self.all_layers_init_conds = [layer.initconds for layer in self.layers]
        self.all_conns_init_conds = [[[conn.source.initconds, conn.matrix] for conn in layer.connections] for layer in self.layers]

    @tf.function
    def integrate(self):

        def odeRK4(all_states, t_idx):

            t = self.time[t_idx] 
            t_plus_half_dt = tf.add(t, self.half_dt)
            t_plus_dt = tf.add(t, self.dt)
            
            all_layers_state, all_conns_state = all_states

            all_layers_k1 = [self.zfun(t, t_idx, 
                                all_layers_state[ilayer], 
                                all_conns_state[ilayer], 
                                self.layers[ilayer].connections, 
                                **self.layers[ilayer].params) 
                            for ilayer in range(len(all_layers_state))]

            all_layers_k2 = [self.zfun(t_plus_half_dt, tf.add(tf.cast(t_idx, dtype=tf.float64),0.5), 
                                tf.add(all_layers_state[ilayer], tf.scalar_mul(self.dt, all_layers_k1[ilayer])),
                                all_conns_state[ilayer], 
                                self.layers[ilayer].connections, 
                                **self.layers[ilayer].params) 
                            for ilayer in range(len(all_layers_state))]

            all_layers_k3 = [self.zfun(t_plus_half_dt, tf.add(tf.cast(t_idx, dtype=tf.float64),0.5), 
                                tf.add(all_layers_state[ilayer], tf.scalar_mul(self.dt, all_layers_k2[ilayer])),
                                all_conns_state[ilayer], 
                                self.layers[ilayer].connections, 
                                **self.layers[ilayer].params) 
                            for ilayer in range(len(all_layers_state))]

            all_layers_k4 = [self.zfun(t_plus_dt, tf.add(t_idx, 1), 
                                tf.add(all_layers_state[ilayer], tf.scalar_mul(self.dt, all_layers_k2[ilayer])),
                                all_conns_state[ilayer], 
                                self.layers[ilayer].connections, 
                                **self.layers[ilayer].params) 
                            for ilayer in range(len(all_layers_state))]

            all_layers_state = [tf.add(all_layers_state[ilayer],
                                    tf.scalar_mul(tf.divide(self.dt, 6), tf.add_n([all_layers_k1[ilayer],
                                                                                tf.scalar_mul(2, all_layers_k2[ilayer]),
                                                                                tf.scalar_mul(2, all_layers_k3[ilayer]),
                                                                                all_layers_k4[ilayer]])))
                                for ilayer in range(len(all_layers_state))]


            #all_conns_state = [[[tf.add(all_conns_state[ilayer][iconn][0], 1), tf.add(all_conns_state[ilayer][iconn][1],1)]
            #                        for iconn in range(len(all_conns_state[ilayer]))] 
            #                    for ilayer in range(len(all_layers_state))]

            return [all_layers_state, all_conns_state]

        all_states = [self.all_layers_init_conds, self.all_conns_init_conds]
        x = tf.scan(odeRK4, tf.range(5), all_states)
        tf.print(len(x))
        tf.print(len(x[0]))
        tf.print(len(x[1]))
        tf.print(len(x[0][0]))
        tf.print(len(x[1][0]))
        tf.print(len(x[1][0][0][0]))
