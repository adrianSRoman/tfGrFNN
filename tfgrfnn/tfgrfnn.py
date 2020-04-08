import numpy as np
import tensorflow as tf
import copy

from ode_functions import xdot_ydot




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
                    zfun = xdot_ydot):

        self.name = name
        self.layers = layers
        self.stim = stim if stim else stimulus()
        self.zfun = zfun
        self.dt = self.stim.dt
        self.half_dt = self.dt/2
        self.nsamps = self.stim.nsamps
        self.dur = self.stim.dur
        self.time = tf.range(self.dur, delta=self.dt, dtype=tf.float32)

    def enumerate_layers(self):
        self.stim.intid = 0
        for ilayer, layer in enumerate(self.layers):
            layer.intid = ilayer+1
        for layer in self.layers:
            for conn in layer.connections:
                conn.sourceintid = conn.source.intid

    def delete_layer_enumeration(self):
        del self.stim.intid 
        for layer in self.layers:
            del layer.intid
            for conn in layer.connections:
                del conn.sourceintid

    def __repr__(self):
        return "<GrFNN Model %s>" % (self.name) 



def get_model_variables_for_integration(Model, dtype=tf.float16):

    Model.enumerate_layers()
    time = tf.cast(Model.time,dtype)
    stim_values = tf.cast(complex2concat(Model.stim.values,2),dtype)
    layers_state = [tf.tile(tf.expand_dims(tf.cast(complex2concat(layer.initconds,0),dtype),axis=0),
                        tf.constant([Model.stim.ndatapoints.numpy(),1])) 
                    for layer in Model.layers]
    layers_alpha = [tf.cast(layer.params['alpha'],dtype) for layer in Model.layers]
    layers_beta1 = [tf.cast(layer.params['beta1'],dtype) for layer in Model.layers]
    layers_beta2 = [tf.cast(layer.params['beta2'],dtype) for layer in Model.layers]
    layers_epsilon = [tf.cast(layer.params['epsilon'],dtype) for layer in Model.layers]
    layers_freqs = [tf.cast(layer.params['freqs'],dtype) for layer in Model.layers]
    layers_conns_source_intid = [[conn.source.intid for conn in layer.connections] 
                    for layer in Model.layers]
    layers_conns_typeint = [[conn.params['typeint'] for conn in layer.connections] 
                    for layer in Model.layers]
    layers_conns_weight = [[tf.cast(conn.params['weight'],dtype) for conn in layer.connections] 
                    for layer in Model.layers]
    layers_connmats = [[tf.cast(complex2concat(conn.matrixinit,0),dtype)
                        for conn in layer.connections]
                    for layer in Model.layers if layer.connections]
    zfun = Model.zfun
    Model.delete_layer_enumeration()

    return layers_state, layers_alpha, layers_beta1, \
            layers_beta2, layers_epsilon, layers_freqs, \
            layers_connmats, layers_conns_source_intid, \
            layers_conns_typeint, layers_conns_weight, \
            zfun, stim_values, time 



def complex2concat(x, axis):
    return tf.concat([tf.math.real(x),tf.math.imag(x)],axis=axis)	



def Runge_Kutta_4(time, layers_state, layers_alpha, layers_beta1,
                layers_beta2, layers_epsilon, layers_freqs,
                layers_connmats, layers_conns_source_intid, 
                layers_conns_typeint, layers_conns_weight, 
                zfun, stim_values, dtype=tf.float16):

    def scan_fn(layers_state, time_dts_stim):
    
        def get_next_k(time_val, layers_state):
    
            layers_k = [zfun(time_val, layer_state, layer_alpha, layer_beta1,
                            layer_beta2, layer_epsilon, layer_freqs, layer_connmats, 
                            layers_state, layer_conns_source_intid, layer_conns_typeint, 
                            layer_conns_weight, dtype)
                for layer_state, layer_alpha, layer_beta1, \
                    layer_beta2, layer_epsilon, layer_freqs, layer_connmats, \
                    layer_conns_source_intid, layer_conns_typeint, \
                    layer_conns_weight \
                in zip(layers_state[1:], layers_alpha, layers_beta1,
                    layers_beta2, layers_epsilon, layers_freqs,
                    layers_connmats, layers_conns_source_intid, 
                    layers_conns_typeint, layers_conns_weight)]
            
            return layers_k
    
        def update_states(time_scaling, layers_k0, layers_k, new_stim):
    
            layers_state = [tf.add(layer_k0, tf.scalar_mul(time_scaling, layer_k)) 
    	                for (layer_k0, layer_k) in zip(layers_k0, layers_k)]
            layers_state.insert(0, new_stim)
    
            return layers_state
    
        t, dt, stim, stim_shift = time_dts_stim
    
        t_plus_half_dt = tf.add(t, dt/2)
        t_plus_dt = tf.add(t, dt)
    
        layers_k0 = layers_state.copy()
        layers_state.insert(0, stim)
    
        layers_k1 = get_next_k(t, layers_state)
        layers_state = update_states(dt/2, layers_k0, layers_k1, 
    				tf.divide(tf.add(stim, stim_shift),2))
        layers_k2 = get_next_k(t_plus_half_dt, layers_state)
        layers_state = update_states(dt/2, layers_k0, layers_k2, 
    				tf.divide(tf.add(stim, stim_shift),2))
        layers_k3 = get_next_k(t_plus_half_dt, layers_state)
        layers_state = update_states(dt, layers_k0, layers_k3, 
    				stim_shift)
        layers_k4 = get_next_k(t_plus_dt, layers_state)
    
        layers_state = [tf.add(layer_k0, 
    		    tf.multiply(dt/6,  tf.add_n([layer_k1,
    						tf.scalar_mul(2, layer_k2),
    						tf.scalar_mul(2, layer_k3),
    						layer_k4])))
                        for (layer_k0, layer_k1, layer_k2, layer_k3, layer_k4) 
                        in zip(layers_k0, layers_k1, layers_k2, layers_k3, layers_k4)]

        return layers_state
    
    dts = time[1:] - time[:-1]
    layers_states = tf.scan(scan_fn, 
    			[time[:-1], 
    			    dts, 
    			    tf.transpose(stim_values[:,:-1,:],(1,0,2)), 
    			    tf.transpose(stim_values[:,1:,:],(1,0,2))], 
    			layers_state)
    
    return layers_states

'''
    def integrate(self, layers_state, layers_connmats_state):
        layers_states, layers_connmats_states = self.odeRK4(layers_state, layers_connmats_state)
        l_GrFNN_r, l_GrFNN_i = tf.split(layers_states[0],2,axis=2) 
        n_GrFNN_r, n_GrFNN_i = tf.split(layers_states[1],2,axis=2) 
        n_GrFNN_abs = tf.sqrt(tf.add(tf.square(n_GrFNN_r), tf.square(n_GrFNN_i)))
        cleaned_r = tf.multiply(l_GrFNN_r, n_GrFNN_abs)
        cleaned = tf.transpose(cleaned_r,(1,2,0))
        cleaned = tf.reduce_mean(cleaned,1)
        cleaned = tf.divide(cleaned, tf.reduce_max(tf.abs(cleaned)))  
        return layers_states, layers_connmats_states, cleaned

    def restore_classes_after_intergration(self, layers_states, layers_connmats_states):
        if self.zfun == xdot_ydot:
            layers_states, layers_connmats_states = self.concat2complex(layers_states, layers_connmats_states)
        self.save_layers_connmats_states(layers_states, layers_connmats_states)
        self.delete_layer_enumeration()
        return self
    
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
'''
