import numpy as np
import tensorflow as tf
import copy

from ode_functions import xdot_ydot, crdot_cidot




def default_neuron_params():

    default_params = {'alpha':tf.constant(0.0, dtype=tf.float32),
                    'beta1':tf.constant(0.0, dtype=tf.float32),
                    'beta2':tf.constant(0.0, dtype=tf.float32),
                    'epsilon':tf.constant(0.0, dtype=tf.float32)}

    return default_params

def default_freqs():

    freqs = tf.constant(np.logspace(np.log10(0.5), np.log10(2.0), 256), 
                        dtype=tf.float32)
    return freqs

class neurons():

    def __init__(self, name = '', 
                    osctype = 'grfnn',
                    params = None,
                    freqs = default_freqs(),
                    initconds = tf.constant(0, dtype=tf.complex64, shape=(256,))):
        
        params = params if params else default_neuron_params()

        self.name = name
        self.osctype = osctype
        self.params = params
        self.initconds = initconds
        self.params['freqs'] = freqs
        self.N = len(freqs)
        self.connections = []

    def __repr__(self):
        return "<Layer with %s %s neurons and %s connections>" % (self.N, 
                                                                    self.osctype, 
                                                                    len(self.connections))




class stimulus():

    def __init__(self, name = '', 
                    values = tf.constant(0, dtype=tf.complex64, shape=(1,1,1)),
                    fs = tf.constant(1.0)):

        self.name = name
        self.values = values
        vshape = tf.shape(self.values)
        self.ndatapoints = vshape[0]
        self.nsamps = vshape[1]
        self.nchannels = vshape[2]
        self.fs = fs
        self.dt = 1.0/self.fs
        self.dur = self.nsamps/self.fs

    def __repr__(self):
        return "<Input stimulus with %s datapoints %s samples and %s channels>" % (self.ndatapoints,
                                                                                    self.nsamps, 
                                                                                    self.nchannels)




class connection():

    def __init__(self, name = '', 
                    source = None,
                    target = None,
                    matrixinit = None,
                    params = None):

        params = params if params else default_connections_params()

        self.name = name
        self.source = source
        self.target = target
        self.params = params
        self.matrixinit = matrixinit
        if isinstance(self.source, stimulus):
            self.params['freqss'] = tf.constant(0, dtype=tf.float32, shape=(self.source.nchannels,))
            self.params['freqst'] = self.target.params['freqs']
        elif isinstance(self.source, neurons):
            self.params['freqss'] = self.source.params['freqs']
            self.params['freqst'] = self.target.params['freqs']
        if self.params['type'] == '1freq':
            self.params['typeint'] = tf.constant(0)
        elif self.params['type'] == 'allfreq':
            self.params['typeint'] = tf.constant(1)

    def __repr__(self):
        return "<Connection from %s to %s with matrix of size %s>" % (self.source.name, 
                                                                        self.target.name, 
                                                                        tf.shape(self.matrixinit).numpy())

def default_connection_params():

    default_params = {'type':'1freq',
                        'learn':False,
                        'lambda_':tf.constant(0.0, dtype=tf.float32),
                        'mu1':tf.constant(0.0, dtype=tf.float32), 
                        'mu2':tf.constant(0.0, dtype=tf.float32), 
                        'epsilon':tf.constant(0.0, dtype=tf.float32),
                        'kappa':tf.constant(0.0, dtype=tf.float32),
                        'weight':tf.constant(1.0, dtype=tf.float32)}

    return default_params

def connect(connname = '', source = None, target = None, matrixinit = None, params = None):

    target.connections = target.connections + \
                            [connection(name = connname,
                                        source=source,
                                        target=target, 
                                        matrixinit=matrixinit, 
                                        params=params)]

    return target
       



class Model():

    def __init__(self, name = '',
                    layers = None,
                    stim = None,
                    zfun = xdot_ydot,
                    cfun = crdot_cidot):

        self.layers = layers
        self.stim = stim if stim else stimulus()
        self.zfun = zfun
        self.cfun = cfun
        self.dt = self.stim.dt
        self.half_dt = self.dt/2
        self.nsamps = self.stim.nsamps
        self.dur = self.stim.dur
        self.time = tf.range(self.dur, delta=self.dt, dtype=tf.float32)

    @tf.function
    def odeRK4(self, layers_state, layers_connmats_state):

        def scan_fn(layers_and_layers_connmats_state, time_dts_stim):

            def get_next_k(time_val, layers_state, layers_connmats_state):

                layers_k = [self.zfun(time_val, layer_state, layer_connmats_state, 
                                    self.layers[ilayer].connections, layers_state, 
                                    **self.layers[ilayer].params)
                                for ilayer, (layer_state, layer_connmats_state) 
                                in enumerate(zip(layers_state[1:], layers_connmats_state))]
                layers_connmats_k = [[self.cfun(time_val,
                                        layers_state[self.layers[ilayer].connections[iconn].sourceintid], 
                                        connmat_state, layer_state, 
                                        self.layers[ilayer].connections[iconn].params)
                                    for iconn, connmat_state in enumerate(layer_connmats_state)]
                                for ilayer, (layer_state, layer_connmats_state) 
                                in enumerate(zip(layers_state[1:], layers_connmats_state)) 
                                if layer_connmats_state]

                return layers_k, layers_connmats_k

            def update_states(time_scaling, layers_k0, layers_k, layers_connmats_k0, layers_connmats_k, new_stim):

                layers_state = [tf.add(layer_k0, tf.scalar_mul(time_scaling, layer_k)) 
                                for (layer_k0, layer_k) in zip(layers_k0, layers_k)]
                layers_connmats_state = [[tf.add(connmat_k0, tf.scalar_mul(time_scaling, connmat_k))
                                    for (connmat_k0, connmat_k) in zip(layer_connmats_k0, layer_connmats_k)]
                                for (layer_connmats_k0, layer_connmats_k) 
                                in zip(layers_connmats_k0, layers_connmats_k) 
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
                            for (layer_k0, layer_k1, layer_k2, layer_k3, layer_k4) 
                            in zip(layers_k0, layers_k1, layers_k2, layers_k3, layers_k4)]
            layers_connmats_state = [[tf.add(connmat_k0, 
                            tf.multiply(dt/6,  tf.add_n([connmat_k1,
                                                        tf.scalar_mul(2, connmat_k2),
                                                        tf.scalar_mul(2, connmat_k3),
                                                        connmat_k4])))
                                for (connmat_k0, connmat_k1, connmat_k2, connmat_k3, connmat_k4) 
                                in zip(layer_connmats_k0, layer_connmats_k1, 
                                    layer_connmats_k2, layer_connmats_k3, layer_connmats_k4)]
                            for (layer_connmats_k0, layer_connmats_k1, 
                                layer_connmats_k2, layer_connmats_k3, layer_connmats_k4) 
                            in zip(layers_connmats_k0, layers_connmats_k1, layers_connmats_k2, 
                                layers_connmats_k3, layers_connmats_k4) 
                            if layer_connmats_k0]

            return [layers_state, layers_connmats_state]

        dts = self.time[1:] - self.time[:-1]
        layers_states, layers_connmats_states = tf.scan(scan_fn, 
                                                        [self.time[:-1], 
                                                            dts, 
                                                            tf.transpose(self.stim.values[:,:-1,:],(1,0,2)), 
                                                            tf.transpose(self.stim.values[:,1:,:],(1,0,2))], 
                                                        [layers_state, layers_connmats_state])

        return layers_states, layers_connmats_states

    def integrate(self):
        self.enumerate_layers()
        if self.zfun == xdot_ydot:
            self.complex2concat()
        layers_state, layers_connmats_state = self.list_layers_state_and_layers_connmats_state()
        layers_states, layers_connmats_states = self.odeRK4(layers_state, layers_connmats_state)
        if self.zfun == xdot_ydot:
            layers_states, layers_connmats_states = self.concat2complex(layers_states, layers_connmats_states)
        self.save_layers_connmats_states(layers_states, layers_connmats_states)
        self.delete_layer_enumeration()
        return self
    
    def delete_layer_enumeration(self):
        del self.stim.intid 
        for layer in self.layers:
            del layer.intid
            for conn in layer.connections:
                del conn.sourceintid

    def save_layers_connmats_states(self, layers_states, layers_connmats_states):
        for ilayer, layer, in enumerate(self.layers):
            layer.states = tf.transpose(layers_states[ilayer],(1,2,0))
            for iconn, conn in enumerate(layer.connections):
                if conn.params['learn']:
                    conn.matrixstates = tf.transpose(layers_connmats_state[ilayer][iconn],(1,2,0))

    def concat2complex(self, layers_states, layers_connmats_states):
        stim_values_real, stim_values_imag = tf.split(self.stim.values, 2, axis=2)
        self.stim.values = tf.complex(stim_values_real, stim_values_imag)
        for ilayer, layer in enumerate(self.layers):
            layer_initconds_real, layer_initconds_imag = tf.split(tf.squeeze(layer.initconds[0,:]), 2, axis=0)
            layer.initconds = tf.complex(layer_initconds_real, layer_initconds_imag)
            layer_states_real, layer_states_imag = tf.split(layers_states[ilayer], 2, axis=2)
            layers_states[ilayer] = tf.complex(layer_states_real, layer_states_imag)
            layer.params['freqs'], _ = tf.split(layer.params['freqs'], 2, axis=0)
            layer.N = layer.N/2
            for iconn, conn in enumerate(layer.connections):
                conn_matrixinit_real, conn_matrixinit_imag = tf.split(conn.matrixinit, 2, axis=1)
                conn.matrixinit = tf.complex(conn_matrixinit_real, conn_matrixinit_imag)
                if conn.params['learn']:
                    connmat_states_real, connmat_states_imag = tf.split(layers_connmats_state[ilayer][iconn], 
                                                                                                    2, axis=2)
                    layers_connmats_states[ilayer][iconn] = tf.complex(connmat_states_real, connmat_states_imag)
        return layers_states, layers_connmats_states

    def list_layers_state_and_layers_connmats_state(self):
        layers_state = [layer.initconds for layer in self.layers]
        layers_connmats_state = [[conn.matrixinit 
                                    for conn in layer.connections] 
                                for layer in self.layers if layer.connections]
        return layers_state, layers_connmats_state

    def complex2concat(self):
        self.stim.values = tf.concat([tf.math.real(self.stim.values), 
                                      tf.math.imag(self.stim.values)], axis=2)
        for layer in self.layers:
            layer.params['freqs'] = tf.concat([layer.params['freqs'], layer.params['freqs']], axis=0)
            layer.initconds = tf.tile(tf.expand_dims(tf.concat([tf.math.real(layer.initconds), 
                                                    tf.math.imag(layer.initconds)], axis=0), axis=0),
                                        tf.constant([self.stim.ndatapoints.numpy(),1], dtype=tf.int32))
            layer.N = layer.N*2
            for conn in layer.connections:
                conn.matrixinit = tf.concat([tf.math.real(conn.matrixinit), 
                                            tf.math.imag(conn.matrixinit)], axis=1)

    def enumerate_layers(self):
        self.stim.intid = 0
        for ilayer, layer in enumerate(self.layers):
            layer.intid = ilayer+1
        for layer in self.layers:
            for conn in layer.connections:
                conn.sourceintid = conn.source.intid
