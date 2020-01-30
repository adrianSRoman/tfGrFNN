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
            self.initconds = tf.constant(0.1, dtype=tf.float64, shape=(self.nosc,))
            self.currstep = tf.constant(0.1, dtype=tf.float64, shape=(self.nosc,))
        else:
            self.initconds = tf.constant(initconds)
            self.currstep = tf.constant(initconds)
        self.savesteps = savesteps
        self.connections = None

    def __repr__(self):
        return "<Layer with %s %s oscillators>" % (self.nosc, self.osctype)

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
        self.half_dt = dt/2
        if self.zfun == xdot_ydot:
            for layer in self.layers:
                layer.params['ones'] = tf.concat([layer.params['ones'], layer.params['ones']], axis=0)
                layer.params['freqs'] = tf.concat([layer.params['freqs'], layer.params['freqs']], axis=0)
                layer.currstep = tf.concat([tf.math.real(layer.currstep), tf.math.imag(layer.currstep)], axis=0)
        for layer in self.layers:
            if layer.savesteps:
                layer.allsteps = []

    @tf.function
    def integrate(self):

        def odeRK4(self, t_idx):
    
            t = self.time[t_idx] 
            t_plus_half_dt = tf.add(t, self.half_dt)
            t_plus_dt = tf.add(t, self.dt)

            for layer in self.layers:
                layer.k0 = layer.currstep
                layer.k1 = self.zfun(t, t_idx, layer.currstep, layer.connections, **layer.params)
                layer.currstep = tf.add(layer.currstep, tf.scalar_mul(self.dt, layer.k1/2))
                if layer.connections != None:
                    for conn in layer.connections:
                        conn.k0 = conn.matrix
                        conn.k1 = self.cfun(t, conn.matrix, conn.params) 
                        conn.matrix = tf.add(conn.matrix, tf.scalar_mul(self.dt, conn.k1/2))

            for layer in self.layers:
                layer.k2 = self.zfun(t_plus_half_dt, tf.cast(t_idx, dtype=tf.float64)+0.5, layer.currstep, layer.connections, **layer.params)
                layer.currstep = tf.add(layer.currstep, tf.scalar_mul(self.dt, layer.k2/2))
                if layer.connections != None:
                    for conn in layer.connections:
                        conn.k2 = self.cfun(t_plust_half_dt, conn.matrix, conn.params) 
                        conn.matrix = tf.add(conn.matrix, tf.scalar_mul(self.dt, conn.k2/2))

            for layer in self.layers:
                layer.k3 = self.zfun(t_plus_half_dt, tf.cast(t_idx, dtype=tf.float64)+0.5, layer.currstep, layer.connections, **layer.params)
                layer.currstep = tf.add(layer.currstep, tf.scalar_mul(self.dt, layer.k3))
                if layer.connections != None:
                    for conn in layer.connections:
                        conn.k3 = self.cfun(t_plus_half_dt, conn.matrix, conn.params) 
                        conn.matrix = tf.add(conn.matrix, tf.scalar_mul(self.dt, conn.k3))
            
            for layer in self.layers:
                layer.k4 = self.zfun(t_plus_dt, t_idx+1, layer.currstep, layer.connections, **layer.params)
                layer.currstep = tf.add(layer.k0, 
                                    tf.scalar_mul(self.dt/6, tf.add_n([layer.k1, 
                                                                        tf.scalar_mul(2, layer.k2), 
                                                                        tf.scalar_mul(2, layer.k3), 
                                                                        layer.k4])))
                tf.print(layer.currstep)
                if layer.connections != None:
                    for conn in layer.connections:
                        conn.k4 = self.cfun(t_plus_dt, conn.matrix, conn.params) 
                        conn.matrix = tf.add(conn.k0, 
                                            tf.scalar_mul(self.dt/6, tf.add_n([conn.k1, 
                                                                                tf.scalar_mul(2, conn.k2), 
                                                                                tf.scalar_mul(2, conn.k3), 
                                                                                conn.k4])))

            return self, tf.add(t_idx, 1)

        t_idx = tf.constant(0)
        tf.while_loop(lambda t_idx: tf.less(t_idx, len(self.time)), odeRK4, [self, t_idx])
