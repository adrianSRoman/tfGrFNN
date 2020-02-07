import numpy as np
import tensorflow as tf
import copy

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
        else:
            self.initconds = tf.constant(initconds)
        self.connections = []

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
                    learnparams = {'learntype':'null',
                                    'lambda_':tf.constant(0.0, dtype=tf.float64), 
                                    'mu1':tf.constant(0.0, dtype=tf.float64), 
                                    'mu2':tf.constant(0.0, dtype=tf.float64), 
                                    'epsilon':tf.constant(0.0, dtype=tf.float64), 
                                    'kappa':tf.constant(0.0, dtype=tf.float64)}):

        self.name = name
        self.source = source
        self.target = target
        self.learnparams = learnparams
        if self.learnparams['learntype'] == '1freq' and isinstance(self.source, stimulus):
            self.matrixinit = tf.complex(tf.random.normal(shape=(self.target.nosc, self.source.nchannels), dtype=tf.float64),
                                    tf.random.normal(shape=(self.target.nosc, self.source.nchannels), dtype=tf.float64))
            self.learnparams['freqss'] = tf.constant(0, dtype=tf.float64, shape=(self.source.nchannels,))
            self.learnparams['freqst'] = self.target.freqs
            self.learnparams['learntypeint'] = tf.constant(0)
        elif self.learnparams['learntype'] == '1freq' and isinstance(self.source, oscillators):
            self.matrixinit = tf.complex(tf.random.normal(shape=(self.target.nosc, self.source.nosc), dtype=tf.float64),
                                    tf.random.normal(shape=(self.target.nosc, self.source.nosc), dtype=tf.float64))
            self.learnparams['freqss'] = self.source.freqs
            self.learnparams['freqst'] = self.target.freqs
            self.learnparams['learntypeint'] = tf.constant(0)


def connect(source=None, target=None, learnparams = {'learntype':'null',
                                                        'lambda_':tf.constant(0.0, dtype=tf.float64), 
                                                        'mu1':tf.constant(0.0, dtype=tf.float64), 
                                                        'mu2':tf.constant(0.0, dtype=tf.float64), 
                                                        'epsilon':tf.constant(0.0, dtype=tf.float64), 
                                                        'kappa':tf.constant(0.0, dtype=tf.float64)}):

    conn = connection(source=source, target=target, learnparams=learnparams)
    target.connections.append(conn) 

    return target
       

class Model():

    def __init__(self, name = '',
                    layers = None,
                    stim = None,
                    zfun = xdot_ydot,
                    cfun = crdot_cidot,
                    time = tf.range(1, delta = 0.01, dtype = tf.float64),
                    dt = tf.constant(0.01, dtype=tf.float64)):

        self.layers = layers
        self.time = time
        self.stim = stim if stim else stimulus(values=tf.constant(0, shape=(1,tf.shape(self.time))))
        self.zfun = zfun
        self.cfun = cfun
        self.dt = dt
        self.half_dt = dt/2

    @tf.function
    def odeRK4(self, layers_and_connmats_state, time_dt_stims):


        def scan_fn(layers_and_connmats_state, time_dts_stims):

            layers_state, connmats_state = layers_and_connmats_state
            t, dt, stim, stim_shift = time_dt_stim

            t_plus_half_dt = tf.add(t, dt/2)
            t_plus_dt = tf.add(t, dt)

            layers_k0 = layers_state.copy()
            connmats_k0 = connmats_state.copy()

            layers_k1 = [self.zfun(t, layer_state, connmats_states[ilayer], 
                            self.layers[ilayer].connections.sourcesintid, 
                            layers_state, **self.layers[ilayer].params)
                        for ilayer, layer_state in enumerate(layers_state)]
            connmats_k1 = [[self.cfun(t, layers_state, connmat_state, 
                                layer_state[self.layers[ilayer].connections[iconn].learnparams['learntypeint'], 
                                self.layer[ilayer].connections[iconn].learnparams) 
                            for iconn, connmat_state in enumerate(connmats_state)] 
                        for ilayer, layer_state in enumerate(layers_state))]
            layers_state = [tf.add(layer_k0, tf.scalar_mul(dt/2, layers_k1[ilayer])) 
                        for ilayer, layer_k0 in enumerate(layers_k0)]
            connmats_state = [tf.add(connmat_k0, tf.scalar_mul(dt/2, connmats_k1[ilayer][ilayer]))
                            for iconn, connmat_k0 in enumerate(connmats_k0[ilayer])
                        for ilayer in range(len(layers_k0))]

            layers_k2 = [self.zfun(t_plus_half_dt, layer_state, connmats_states[ilayer], 
                            self.layers[ilayer].connections.sourcesintid, 
                            layers_state, **self.layers[ilayer].params)
                        for ilayer, layer_state in enumerate(layers_state)]
            connmats_k2 = [[self.cfun(t_plus_half_dt, layers_state, connmat_state, 
                                layer_state[self.layers[ilayer].connections[iconn].learnparams['learntypeint'], 
                                self.layer[ilayer].connections[iconn].learnparams) 
                            for iconn, connmat_state in enumerate(connmats_state)] 
                        for ilayer, layer_state in enumerate(layers_state))]
            layers_state = [tf.add(layer_k0, tf.scalar_mul(dt/2, layers_k2[ilayer])) 
                        for ilayer, layer_k0 in enumerate(layers_k0)]
            connmats_state = [tf.add(connmat_k0, tf.scalar_mul(dt/2, connmats_k2[ilayer][ilayer]))
                            for iconn, connmat_k0 in enumerate(connmats_k0[ilayer])
                        for ilayer in range(len(layers_k0))]

            layers_k3 = [self.zfun(t_plus_half_dt, layer_state, connmats_states[ilayer], 
                            self.layers[ilayer].connections.sourcesintid, 
                            layers_state, **self.layers[ilayer].params)
                        for ilayer, layer_state in enumerate(layers_state)]
            connmats_k3 = [[self.cfun(t_plus_half_dt, layers_state, connmat_state, 
                                layer_state[self.layers[ilayer].connections[iconn].learnparams['learntypeint'], 
                                self.layer[ilayer].connections[iconn].learnparams) 
                            for iconn, connmat_state in enumerate(connmats_state)] 
                        for ilayer, layer_state in enumerate(layers_state))]
            layers_state = [tf.add(layer_k0, tf.scalar_mul(dt, layers_k2[ilayer])) 
                        for ilayer, layer_k0 in enumerate(layers_k0)]
            connmats_state = [tf.add(connmat_k0, tf.scalar_mul(dt, connmats_k2[ilayer][ilayer]))
                            for iconn, connmat_k0 in enumerate(connmats_k0[ilayer])
                        for ilayer in range(len(layers_k0))]

            layers_k4 = [self.zfun(t_plus_dt, layer_state, connmats_states[ilayer], 
                            self.layers[ilayer].connections.sourcesintid, 
                            layers_state, **self.layers[ilayer].params)
                        for ilayer, layer_state in enumerate(layers_state)]
            connmats_k4 = [[self.cfun(t_plus_dt, layers_state, connmat_state, 
                                layer_state[self.layers[ilayer].connections[iconn].learnparams['learntypeint'], 

            layers_state = [tf.add(layer_k0, 
                            tf.multiply(dt/6,  tf.add_n([layers_k1[ilayer],
                                                        tf.scalar_mul(2, layers_k2[ilayer]),
                                                        tf.scalar_mul(2, layers_k3[ilayer]),
                                                        layers_k4[ilayer]])))
                        for ilayer, layer_k0 in enumerate(layers_k0)]
            connmats_state = [[tf.add(conn_k0, 
                            tf.multiply(dt/6,  tf.add_n([connmats_k1[ilayer][iconn],
                                                        tf.scalar_mul(2, connmats_k2[ilayer][iconn]),
                                                        tf.scalar_mul(2, connmats_k3[ilayer][iconn]),
                                                        connmats_k4[ilayer][iconn]])))]
                            for iconn, conn_k0 in enumerate(connmats_k0[ilayer])
                        for ilayer in range(len(layers_k0))]

            return [layers_state, connmats_state]

        dts = self.time[1:] - self.time[:-1]
        layers_state, connmats_state = tf.scan(scan_fn, 
                    [self.time[:-1], dts, self.stim.values[:-1], self.stim.values[1:]], 
                    [layers_state, connmats_state])

        return layers_state, connmats_state

    def integrate(self):

        if self.zfun == xdot_ydot:
            for layer in self.layers:
                layer.params['freqs'] = tf.concat([layer.params['freqs'], layer.params['freqs']], axis=0)
                layer.initconds = tf.concat([tf.math.real(layer.initconds), tf.math.imag(layer.initconds)], axis=0)
                layer.nosc = layer.nosc*2
                for conn in layer.connections:
                    conn.matrixinit = tf.concat([tf.math.real(conn.matrixinit), tf.math.imag(conn.matrixinit)], axis=0)
            for stim in self.stimuli:
                stim.values = tf.concat([tf.math.real(stim.values), tf.math.imag(stim.values)], axis=0)

        layers_state = [layer.currstate for layer in self.layers]
        connmats_state = [[conn.matrixstate for conn in layers.connections] for layer in layers]

        layers_state, connmats_state = self.odeRK4(layers_state, connmats_state)

        if self.zfun == xdot_ydot:
            for stim in self.stimuli:
                stim_values_real, stim_values_imag = tf.split(stim.values, 2, axis=0)
                stim.values = tf.complex(stim_values_real, stim_values_imag)
            for layer in self.layers:
                layer_initconds_real, layer_initconds_imag = tf.split(layer.initconds, 2, axis=0)
                layer.initconds = tf.complex(layer_initconds_real, layer_initconds_imag)
                layer.params['freqs'], _ = tf.split(layer.params['freqs'], 2, axis=0)
                layer.nosc = layer.nosc/2
                for conn in layer.connections:
                    conn_matrixinit_real, conn_matrixinit_imag = tf.split(conn.matrixinit, 2, axis=0)
                    conn.matrixinit = tf.complex(conn_matrixinit_real, conn_matrixinit_imag)
       
        return self
