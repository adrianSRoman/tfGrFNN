import numpy as np
import tensorflow as tf
import time

from oscillator_types import canonical_hopf
from connection_learning_types import learn_1freq 
from ode_functions import xdot_ydot, crdot_cidot


class oscillators():

    def __init__(self, name = '', 
                    osctype = canonical_hopf(),
                    nosc = 256,
                    freqlims = (0.12, 8.0),
                    freqspacing = 'log',
                    initconds = None):

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
        if initconds == None:
            self.initconds = tf.constant(0.99, dtype=tf.complex128, shape=(self.nosc,))
            self.currstate = self.initconds
        else:
            self.initconds = tf.constant(initconds)
            self.currstate = self.initconds
        self.connections = [connection(source=self, target=self, learnparams={'learntype':'null'})]

    def __repr__(self):
        return "<Layer with %s %s oscillators and %s connections>" % (self.nosc, self.osctype, len(self.connections))


class stimulus():

    def __init__(self, name='', 
                    values = tf.constant(0, dtype=tf.complex128, shape=(1,1)),
                    fs = tf.constant(1.0)):

        self.values = values
        self.shape = tf.shape(self.values)
        self.nchannels = self.shape[0]
        self.ntime = self.shape[1]
        self.fs = fs
        self.dt = 1.0/self.fs
        self.initconds = self.values[:,0]
        if tf.shape(self.values)[1] == 1:
            self.stimtype = 'null'
        else:
            self.stimtype = 'values'


class connection():

    def __init__(self, name = '', 
                    source = None,
                    target = None,
                    learnparams = {'learntype':'null'}):

        self.name = name
        self.source = source
        self.target = target
        self.learnparams = learnparams
        if self.learnparams['learntype'] == 'null':
            self.typeint = tf.constant(0)
            self.matrix = tf.constant(0, dtype=tf.complex128, shape=(1, self.source.nosc))
            self.learnparams['learntypeint'] = tf.constant(0)
        elif self.learnparams['learntype'] == '1freq' and isinstance(self.source, stimulus):
            self.matrix = tf.complex(tf.random.normal(shape=(self.target.nosc, self.source.nchannels), dtype=tf.float64),
                                    tf.random.normal(shape=(self.target.nosc, self.source.nchannels), dtype=tf.float64))
            self.learnparams['freqss'] = self.target.freqs
            self.learnparams['freqst'] = self.target.freqs
            self.learnparams['learntypeint'] = tf.constant(1)
        elif self.learnparams['learntype'] == '1freq' and isinstance(self.source, oscillators):
            self.matrix = tf.complex(tf.random.normal(shape=(self.target.nosc, self.source.nosc), dtype=tf.float64),
                                    tf.random.normal(shape=(self.target.nosc, self.source.nosc), dtype=tf.float64))
            self.learnparams['freqss'] = self.source.freqs
            self.learnparams['freqst'] = self.target.freqs
            self.learnparams['learntypeint'] = tf.constant(1)


def connect(source=None, target=None, learnparams={'learntype':'null'}):

    conn = connection(source=source, target=target, learnparams=learnparams)
    target.connections.append(conn) 

    return target
       

class Model():

    def __init__(self, name = '',
                    layers = None,
                    stimuli = [stimulus()],
                    zfun = xdot_ydot,
                    cfun = crdot_cidot,
                    time = tf.range(1, delta = 0.01, dtype = tf.float64),
                    dt = tf.constant(0.01, dtype=tf.float64)):

        self.layers = layers
        self.stimuli = stimuli
        self.time = time
        self.zfun = zfun
        self.cfun = cfun
        self.dt = dt
        self.half_dt = dt/2
        for layer in self.layers:
            layer.allsteps = tf.constant(0.0, 
                                        dtype=tf.float64, 
                                        shape=(tf.shape(layer.initconds)[0], tf.shape(time)[0]))
            for conn in layer.connections:
                conn.matrix = tf.constant(0.0, 
                                        dtype=tf.float64, 
                                        shape=(tf.shape(conn.matrix)[0], 
                                            tf.shape(conn.matrix)[1], 
                                            tf.shape(time)[0]))

    @tf.function
    def odeRK4(self):

        def RK(t_idx):

            t = self.time[t_idx] 
            t_plus_half_dt = tf.add(t, self.half_dt)
            t_plus_dt = tf.add(t, self.dt)
           
            for layer in self.layers:
                layer.k0 = layer.currstate
                layer.k1 = self.zfun(t, t_idx, 
                                layer.currstate,
                                layer.connections,
                                **layer.params) 
                for conn in layer.connections:
                    conn.matrix_k0 = conn.matrix
                    conn.matrix_k1 = self.cfun(t, t_idx,
                                            conn.source.currstate,
                                            conn.matrix,
                                            conn.target.currstate,
                                            conn.learnparams)
            for layer in self.layers:
                layer.currstate = tf.add(layer.k0, tf.scalar_mul(self.half_dt, layer_k1))
                for conn in layer.connections:
                    conn.matrix = tf.add(conn.matrix_k0, tf.scalar_mul(self.half_dt, conn.matrix_k1))

            for layer in self.layers:
                layer.k2 = self.zfun(t_plus_half_dt, tf.add(tf.cas(t_idx, dtype=tf.float64), 0.5), 
                                layer.currstate,
                                layer.connections,
                                **layer.params) 
                for conn in layer.connections:
                    conn.matrix_k1 = self.cfun(t_plus_half_dt, tf.add(tf.cas(t_idx, dtype=tf.float64), 0.5), 
                                            conn.source.currstate,
                                            conn.matrix,
                                            conn.target.currstate,
                                            conn.learnparams)
            for layer in self.layers:
                layer.currstate = tf.add(layer.k0, tf.scalar_mul(self.half_dt, layer_k2))
                for conn in layer.connections:
                    conn.matrix = tf.add(conn.matrix_k0, tf.scalar_mul(self.half_dt, conn.matrix_k2))

            for layer in self.layers:
                layer.k3 = self.zfun(t_plus_half_dt, tf.add(tf.cas(t_idx, dtype=tf.float64), 0.5), 
                                layer.currstate,
                                layer.connections,
                                **layer.params) 
                for conn in layer.connections:
                    conn.matrix_k1 = self.cfun(t_plus_half_dt, tf.add(tf.cas(t_idx, dtype=tf.float64), 0.5), 
                                            conn.source.currstate,
                                            conn.matrix,
                                            conn.target.currstate,
                                            conn.learnparams)
            for layer in self.layers:
                layer.currstate = tf.add(layer.k0, layer_k3)
                for conn in layer.connections:
                    conn.matrix = tf.add(conn.matrix_k0, conn.matrix_k3)

            for layer in self.layers:
                layer.k4 = self.zfun(t_plus_dt, tf.add(t_idx, 1), 
                                layer.currstate,
                                layer.connections,
                                **layer.params) 
                for conn in layer.connections:
                    conn.matrix_k4 = self.cfun(t_plus_dt, tf.add(t_idx, 1), 
                                            conn.source.currstate,
                                            conn.matrix,
                                            conn.target.currstate,
                                            conn.learnparams)

            for layer in self.layers:
                layer.currstate = tf.add(layer.k0, 
                                    tf.scalar_mul(tf.divide(self.dt, 6),
                                        tf.add_n(layer.k1, 
                                                    tf.scalar_mul(2, layer.k2),
                                                    tf.scalar_mul(2, layer.k3),
                                                    layer.k4)))

            return tf.add(t_idx, 1)

        t_idx = 0
        tf.while_loop(lambda t_idx: tf.less(t_idx,tf.shape(self.time)), RK, [t_idx])
        
        return all_layers_steps, all_conns_steps

    def integrate(self):

        if self.zfun == xdot_ydot:
            for layer in self.layers:
                layer.params['freqs'] = tf.concat([layer.params['freqs'], layer.params['freqs']], axis=0)
                layer.initconds = tf.concat([tf.math.real(layer.initconds), tf.math.imag(layer.initconds)], axis=0)
                layer.currstate = layer.initconds
                layer.nosc = layer.nosc*2
                for conn in layer.connections:
                    conn.matrix = tf.concat([tf.math.real(conn.matrix), tf.math.imag(conn.matrix)], axis=0)
            for stim in self.stimuli:
                stim.values = tf.concat([tf.math.real(stim.values), tf.math.imag(stim.values)], axis=0)
                stim.initconds = tf.concat([tf.math.real(stim.initconds), tf.math.imag(stim.initconds)], axis=0)
                stim.currstate = stim.initconds

        self.odeRK4()

        if self.zfun == xdot_ydot:
            for stim in self.stimuli:
                stim_values_real, stim_values_imag = tf.split(stim.values, 2, axis=0)
                stim.values = tf.complex(stim_values_real, stim_values_imag)
                stim_initconds_real, stim_initconds_imag = tf.split(stim.initconds, 2, axis=0)
                stim.initconds = tf.complex(stim_initconds_real, stim_initconds_imag)
            for layer in self.layers:
                layer.params['freqs'], _ = tf.split(layer.params['freqs'], 2, axis=0)
                initconds_real, initconds_imag = tf.split(layer.initconds, 2, axis=0)
                layer.initconds = tf.complex(initconds_real, initconds_imag)
                layer.nosc = layer.nosc/2
                layer_steps_real, layer_steps_imag = tf.split(tf.transpose(layers.allsteps), 2, axis=0)
                layer.allsteps = tf.complex(layer_steps_real, layer_steps_imag)
                for conn in layer.connections:
                    conn_matrix_real, conn_matrix_imag = tf.split(conn.matrix, 2, axis=0)
                    conn.matrix = tf.complex(conn_matrix_real, conn_matrix_imag)
       
        return self
