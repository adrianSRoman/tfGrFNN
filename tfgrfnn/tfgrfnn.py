import numpy as np
import tensorflow as tf
import copy

from oscillator_types import canonical_hopf
from ode_functions import xdot_ydot, crdot_cidot


class oscillators():

    def __init__(self, name = '', 
                    osctype = canonical_hopf().params(),
                    nosc = 256,
                    freqlims = (0.12, 8.0),
                    freqspacing = 'log',
                    initconds = tf.constant(0, dtype=tf.complex128, shape=(256,))):

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
        self.params = self.osctype
        self.params['freqs'] = tf.constant(self.freqs, dtype=tf.float64)
        self.initconds = tf.constant(initconds)
        self.connections = []

    def __repr__(self):
        return "<Layer with %s %s oscillators and %s connections>" % (self.nosc, self.osctype, len(self.connections))


class stimulus():

    def __init__(self, name='', 
                    values = tf.constant(0, dtype=tf.complex128),
                    fs = tf.constant(1.0)):

        self.name = name
        self.values = values
        self.shape = tf.shape(self.values)
        self.ntime = self.shape[0]
        self.nchannels = self.shape[1] if len(self.shape) > 1 else tf.constant(1)
        self.fs = fs
        self.dt = 1.0/self.fs


class connection():

    def __init__(self, name = '', 
                    source = None,
                    target = None,
                    matrixinit = 1.0+1j*1.0,
                    learnparams = None):

        self.name = name
        self.source = source
        self.target = target
        self.learnparams = learnparams if learnparams else {'learntype':'nolearning',
                                    'lambda_':tf.constant(0.0, dtype=tf.float64), 
                                    'mu1':tf.constant(0.0, dtype=tf.float64), 
                                    'mu2':tf.constant(0.0, dtype=tf.float64), 
                                    'epsilon':tf.constant(0.0, dtype=tf.float64), 
                                    'kappa':tf.constant(0.0, dtype=tf.float64),
                                    'weight':tf.constant(1.0, dtype=tf.float64)}
        self.matrixinit = matrixinit
        if self.learnparams['learntype'] == 'nolearning' and isinstance(self.source, stimulus):
            self.matrixinit = tf.constant(self.matrixinit, dtype=tf.complex128, shape=(self.target.nosc, self.source.nchannels))
            self.learnparams['freqss'] = tf.constant(0, dtype=tf.float64, shape=(self.source.nchannels,))
            self.learnparams['freqst'] = tf.constant(0, dtype=tf.float64, shape=(self.target.nosc,))
            self.learnparams['learntypeint'] = tf.constant(0)
        elif self.learnparams['learntype'] == 'nolearning' and isinstance(self.source, oscillators):
            self.matrixinit = tf.constant(self.matrixinit, dtype=tf.complex128, shape=(self.target.nosc, self.source.nosc))
            self.learnparams['freqss'] = tf.constant(0, dtype=tf.float64, shape=(self.source.nosc,))
            self.learnparams['freqst'] = tf.constant(0, dtype=tf.float64, shape=(self.target.nosc,))
            self.learnparams['learntypeint'] = tf.constant(0)
        elif self.learnparams['learntype'] == '1freq' and isinstance(self.source, stimulus):
            self.matrixinit = tf.complex(tf.random.normal(shape=(self.target.nosc, self.source.nchannels), dtype=tf.float64),
                                    tf.random.normal(shape=(self.target.nosc, self.source.nchannels), dtype=tf.float64))
            self.learnparams['freqss'] = tf.constant(0, dtype=tf.float64, shape=(self.source.nchannels,))
            self.learnparams['freqst'] = self.target.freqs
            self.learnparams['learntypeint'] = tf.constant(1)
        elif self.learnparams['learntype'] == '1freq' and isinstance(self.source, oscillators):
            self.matrixinit = tf.complex(tf.random.normal(shape=(self.target.nosc, self.source.nosc), dtype=tf.float64, stddev=0.01),
                                    tf.random.normal(shape=(self.target.nosc, self.source.nosc), dtype=tf.float64, stddev=0.01))
            self.learnparams['freqss'] = self.source.freqs
            self.learnparams['freqst'] = self.target.freqs
            self.learnparams['learntypeint'] = tf.constant(1)


def connect(source=None, target=None, matrixinit=1.0+1j*1.0,  learnparams=None):

    target.connections = target.connections + [connection(source=source, target=target, matrixinit=matrixinit, learnparams=learnparams if learnparams else {'learntype':'nolearning',
                                        'lambda_':tf.constant(0.0, dtype=tf.float64), 
                                        'mu1':tf.constant(0.0, dtype=tf.float64), 
                                        'mu2':tf.constant(0.0, dtype=tf.float64), 
                                        'epsilon':tf.constant(0.0, dtype=tf.float64), 
                                        'kappa':tf.constant(0.0, dtype=tf.float64),
                                        'weight':tf.constant(1.0, dtype=tf.float64)})]

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
        self.dt = dt
        self.half_dt = dt/2
        self.stim = stim if stim else stimulus(values=tf.constant(0, dtype=tf.float64, shape=(tf.shape(self.time))), fs=1/self.dt)
        self.zfun = zfun
        self.cfun = cfun

    @tf.function
    def odeRK4(self, layers_state, layers_connmats_state):

        def scan_fn(layers_and_layers_connmats_state, time_dts_stim):

            def get_next_k(time_val, layers_state, layers_connmats_state):

                layers_k = [self.zfun(time_val, layer_state, layer_connmats_state, 
                                self.layers[ilayer].connections, layers_state, **self.layers[ilayer].params)
                                for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state))]
                layers_connmats_k = [[self.cfun(time_val, layers_state[self.layers[ilayer].connections[iconn].sourceintid], 
                                    connmat_state, layer_state, self.layers[ilayer].connections[iconn].learnparams)
                                    for iconn, connmat_state in enumerate(layer_connmats_state)]
                                for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state)) 
                                if layer_connmats_state]

                return layers_k, layers_connmats_k

            def update_states(time_scaling, layers_k0, layers_k, layers_connmats_k0, layers_connmats_k, new_stim):

                layers_state = [tf.add(layer_k0, tf.scalar_mul(time_scaling, layer_k)) 
                                for (layer_k0, layer_k) in zip(layers_k0, layers_k)]
                layers_connmats_state = [[tf.add(connmat_k0, tf.scalar_mul(time_scaling, connmat_k))
                                    for (connmat_k0, connmat_k) in zip(layer_connmats_k0, layer_connmats_k)]
                                for (layer_connmats_k0, layer_connmats_k) in zip(layers_connmats_k0, layers_connmats_k) 
                                if layer_connmats_k0]
                layers_state.insert(0, new_stim)

                return layers_state, layers_connmats_state

            layers_state, layers_connmats_state = layers_and_layers_connmats_state
            t, dt, stim, stim_shift = time_dts_stim

            t_plus_half_dt = tf.add(t, dt/2)
            t_plus_dt = tf.add(t, dt)

            layers_k0 = layers_state.copy()
            layers_state.insert(0, stim)
            layers_connmats_k0 = layers_connmats_state.copy()

            layers_k1, layers_connmats_k1 = get_next_k(t, layers_state, layers_connmats_state)
            layers_state, layers_connmats_state = update_states(dt/2, layers_k0, layers_k1, 
                                                                layers_connmats_k0, layers_connmats_k1, 
                                                                tf.divide(tf.add(stim, stim_shift),2))
            layers_k2, layers_connmats_k2 = get_next_k(t_plus_half_dt, layers_state, layers_connmats_state)
            layers_state, layers_connmats_state = update_states(dt/2, layers_k0, layers_k2, 
                                                                layers_connmats_k0, layers_connmats_k2, 
                                                                tf.divide(tf.add(stim, stim_shift),2))
            layers_k3, layers_connmats_k3 = get_next_k(t_plus_half_dt, layers_state, layers_connmats_state)
            layers_state, layers_connmats_state = update_states(dt, layers_k0, layers_k3, 
                                                                layers_connmats_k0, layers_connmats_k3, 
                                                                stim_shift)
            layers_k4, layers_connmats_k4 = get_next_k(t_plus_dt, layers_state, layers_connmats_state)

            layers_state = [tf.add(layer_k0, 
                            tf.multiply(dt/6,  tf.add_n([layer_k1,
                                                        tf.scalar_mul(2, layer_k2),
                                                        tf.scalar_mul(2, layer_k3),
                                                        layer_k4])))
                            for (layer_k0, layer_k1, layer_k2, layer_k3, layer_k4) in zip(layers_k0, layers_k1, layers_k2, layers_k3, layers_k4)]
            layers_connmats_state = [[tf.add(connmat_k0, 
                            tf.multiply(dt/6,  tf.add_n([connmat_k1,
                                                        tf.scalar_mul(2, connmat_k2),
                                                        tf.scalar_mul(2, connmat_k3),
                                                        connmat_k4])))
                                for (connmat_k0, connmat_k1, connmat_k2, connmat_k3, connmat_k4) in zip(layer_connmats_k0, layer_connmats_k1, layer_connmats_k2, layer_connmats_k3, layer_connmats_k4)]
                            for (layer_connmats_k0, layer_connmats_k1, layer_connmats_k2, layer_connmats_k3, layer_connmats_k4) in zip(layers_connmats_k0, layers_connmats_k1, layers_connmats_k2, layers_connmats_k3, layers_connmats_k4) if layer_connmats_k0]

            return [layers_state, layers_connmats_state]

        dts = self.time[1:] - self.time[:-1]
        layers_state, layers_connmats_state = tf.scan(scan_fn, 
                    [self.time[:-1], dts, self.stim.values[:-1], self.stim.values[1:]], 
                    [layers_state, layers_connmats_state])

        return layers_state, layers_connmats_state

    def integrate(self):

        for ilayer, layer in enumerate(self.layers):
            layer.intid = ilayer+1

        if self.zfun == xdot_ydot:
            self.stim.values = tf.concat([tf.expand_dims(tf.math.real(self.stim.values),-1), 
                                tf.expand_dims(tf.math.imag(self.stim.values),-1)], axis=1)
            self.stim.intid = 0
            for layer in self.layers:
                layer.params['freqs'] = tf.concat([layer.params['freqs'], layer.params['freqs']], axis=0)
                layer.initconds = tf.concat([tf.math.real(layer.initconds), tf.math.imag(layer.initconds)], axis=0)
                layer.nosc = layer.nosc*2
                for conn in layer.connections:
                    conn.sourceintid = conn.source.intid
                    conn.matrixinit = tf.concat([tf.math.real(conn.matrixinit), tf.math.imag(conn.matrixinit)], axis=0)

        layers_state = [layer.initconds for layer in self.layers]
        layers_connmats_state = [[conn.matrixinit 
                                    for conn in layer.connections] 
                                for layer in self.layers if layer.connections]

        layers_state, layers_connmats_state = self.odeRK4(layers_state, layers_connmats_state)

        if self.zfun == xdot_ydot:
            stim_values_real, stim_values_imag = tf.split(self.stim.values, 2, axis=1)
            self.stim.values = tf.squeeze(tf.complex(stim_values_real, stim_values_imag))
            del self.stim.intid
            for ilayer, layer in enumerate(self.layers):
                layer_allsteps_real, layer_allsteps_imag = tf.split(layers_state[ilayer], 2, axis=1)
                layer.allsteps = tf.squeeze(tf.complex(layer_allsteps_real, layer_allsteps_imag))
                layer_initconds_real, layer_initconds_imag = tf.split(layer.initconds, 2, axis=0)
                layer.initconds = tf.complex(layer_initconds_real, layer_initconds_imag)
                layer.params['freqs'], _ = tf.split(layer.params['freqs'], 2, axis=0)
                layer.nosc = layer.nosc/2
                del layer.intid
                for iconn, conn in enumerate(layer.connections):
                    conn_matrixinit_real, conn_matrixinit_imag = tf.split(conn.matrixinit, 2, axis=0)
                    conn.matrixinit = tf.complex(conn_matrixinit_real, conn_matrixinit_imag)
                    if conn.learnparams['learntypeint'] == 0:
                        conn.allmatrixsteps = []
                    else:
                        connmat_states_real, connmat_states_imag = tf.split(layers_connmats_state[ilayer][iconn], 2, axis=1)
                        conn.allmatrixsteps = tf.complex(connmat_states_real, connmat_states_imag)
                    del conn.sourceintid
       
        del layers_state
        del layers_connmats_state

        return self
