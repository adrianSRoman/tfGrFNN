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
        self.ntime = self.shape[0]
        self.nchannels = self.shape[1] if len(self.shape) > 1 else tf.constant(1)
        self.fs = fs
        self.dt = 1.0/self.fs


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
    def odeRK4(self, layers_state, layers_connmats_state):


        def scan_fn(layers_and_layers_connmats_state, time_dts_stim):

            layers_state, layers_connmats_state = layers_and_layers_connmats_state
            t, dt, stim, stim_shift = time_dts_stim

            t_plus_half_dt = tf.add(t, dt/2)
            t_plus_dt = tf.add(t, dt)

            layers_k0 = layers_state.copy()
            layers_state.insert(0, stim)
            layers_connmats_k0 = layers_connmats_state.copy()


            layers_k1 = [self.zfun(t, layer_state, layer_connmats_state, 
                            self.layers[ilayer].connections, layers_state, **self.layers[ilayer].params)
                            for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state))]
            layers_connmats_k1 = [[self.cfun(t, layers_state[self.layers[ilayer].connections[iconn].sourceintid], 
                                connmat_state, layer_state, self.layers[ilayer].connections[iconn].learnparams)
                                for iconn, connmat_state in enumerate(layer_connmats_state)] 
                            for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state)) 
                            if layer_connmats_state]
            layers_state = [tf.add(layer_k0, tf.scalar_mul(dt/2, layer_k1)) 
                            for (layer_k0, layer_k1) in zip(layers_k0, layers_k1)]
            layers_connmats_state = [[tf.add(connmat_k0, tf.scalar_mul(dt/2, connmat_k1))
                                for (connmat_k0, connmat_k1) in zip(layer_connmats_k0, layer_connmats_k1)]
                            for (layer_connmats_k0, layer_connmats_k1) in zip(layers_connmats_k0, layers_connmats_k1) if layer_connmats_k0]
            layers_state.insert(0, tf.divide(tf.add(stim, stim_shift), 2))

            layers_k2 = [self.zfun(t_plus_half_dt, layer_state, layer_connmats_state, 
                            self.layers[ilayer].connections, layers_state, **self.layers[ilayer].params)
                            for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state))]
            layers_connmats_k2 = [[self.cfun(t_plus_half_dt, layers_state[self.layers[ilayer].connections[iconn].sourceintid], 
                                connmat_state, layer_state, self.layers[ilayer].connections[iconn].learnparams)
                                for iconn, connmat_state in enumerate(layer_connmats_state)] 
                            for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state)) if layer_connmats_state]
            layers_state = [tf.add(layer_k0, tf.scalar_mul(dt/2, layer_k2)) 
                            for (layer_k0, layer_k2) in zip(layers_k0, layers_k2)]
            layers_connmats_state = [[tf.add(connmat_k0, tf.scalar_mul(dt/2, connmat_k2))
                                for (connmat_k0, connmat_k2) in zip(layer_connmats_k0, layer_connmats_k2)]
                            for (layer_connmats_k0, layer_connmats_k2) in zip(layers_connmats_k0, layers_connmats_k2) if layer_connmats_k0]
            layers_state.insert(0, tf.divide(tf.add(stim, stim_shift), 2))

            layers_k3 = [self.zfun(t_plus_half_dt, layer_state, layer_connmats_state, 
                            self.layers[ilayer].connections, layers_state, **self.layers[ilayer].params)
                            for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state))]
            layers_connmats_k3 = [[self.cfun(t_plus_half_dt, layers_state[self.layers[ilayer].connections[iconn].sourceintid], 
                                connmat_state, layer_state, self.layers[ilayer].connections[iconn].learnparams)
                                for iconn, connmat_state in enumerate(layer_connmats_state)] 
                            for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state)) if layer_connmats_state]
            layers_state = [tf.add(layer_k0, tf.scalar_mul(dt, layer_k3)) 
                            for (layer_k0, layer_k3) in zip(layers_k0, layers_k3)]
            layers_connmats_state = [[tf.add(connmat_k0, tf.scalar_mul(dt, connmat_k3))
                                for (connmat_k0, connmat_k3) in zip(layer_connmats_k0, layer_connmats_k3)]
                            for (layer_connmats_k0, layer_connmats_k3) in zip(layers_connmats_k0, layers_connmats_k3) if layer_connmats_k0]
            layers_state.insert(0, stim_shift)

            layers_k4 = [self.zfun(t_plus_dt, layer_state, layer_connmats_state, 
                            self.layers[ilayer].connections, layers_state, **self.layers[ilayer].params)
                            for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state))]
            layers_connmats_k4 = [[self.cfun(t_plus_dt, layers_state[self.layers[ilayer].connections[iconn].sourceintid], 
                                connmat_state, layer_state, self.layers[ilayer].connections[iconn].learnparams)
                                for iconn, connmat_state in enumerate(layer_connmats_state)] 
                            for ilayer, (layer_state, layer_connmats_state) in enumerate(zip(layers_state[1:], layers_connmats_state)) if layer_connmats_state]

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
                    conn.matrixinit = tf.concat([tf.math.real(conn.matrixinit), tf.math.imag(conn.matrixinit)], axis=0)
                    conn.sourceintid = conn.source.intid

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
                layer_initconds_real, layer_initconds_imag = tf.split(layer.initconds, 2, axis=0)
                layer.initconds = tf.complex(layer_initconds_real, layer_initconds_imag)
                layer.params['freqs'], _ = tf.split(layer.params['freqs'], 2, axis=0)
                layer.nosc = layer.nosc/2
                layer.allsteps = layers_state[ilayer]
                del layer.intid
                for iconn, conn in enumerate(layer.connections):
                    conn_matrixinit_real, conn_matrixinit_imag = tf.split(conn.matrixinit, 2, axis=0)
                    conn.matrixinit = tf.complex(conn_matrixinit_real, conn_matrixinit_imag)
                    conn.allmatrixsteps = layers_connmats_state[ilayer][iconn]
                    del conn.sourceintid
       
        return self
