import numpy as np
import tensorflow as tf

from oscillator_types import hopf
from connection_learning_rule_types import phase_lock 

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
                    time = tf.range(1, 0.01, dtype = tf.float64),
                    dt = 0.01):

        self.layers = layers
        self.time = time
        self.zfun = zfun
        self.cfun = cfun
        self.dt = dt

    def integrate(self):

        all_osc_init_conds, all_conn_init_conds = [oscillatos.initconds, [connection.initconds for connection in oscillators] 
                                                    for oscillators in self.layers]

        def scan_func(all_init_conds, time):

            all_osc_init_conds, all_conn_init_conds = all_init_conds

            all_osc_stepsi, all_conn_steps = [self.odeRK4(self.zfun, time, self.dt, osc_init_conds, M.layers[iosc].params)
                                                [self.odeRK4(self.cfun, time, self.dt, conn_init_conds, M.layers[iconn].params)
                                                    for iconn, conn_init_conds in enumerate(all_conn_init_conds[ilayer])]
                                                for iosc, osc_init_conds in enumerate(all_osc_init_conds)]  
            all_conn_steps = [self.odeRK4(self.cfun, time, self.dt, conn_init_conds) for conn_init_conds in all_conn_init_conds]  

            return all_osc_steps, all_conn_steps

        all_osc_and_conn_steps = tf.scan(scan_func, time, (all_osc_init_conds, all_conn_init_conds))
        
        all_osc_steps, all_conn_steps = all_osc_and_conn_steps

        for ilayer in len(self.layers):
            self.layers[ilayer].savedStates = all_osc_steps[ilayer]
            for iconn in len(self.layers[ilayer].connections):
                self.layers[ilayer].connections[iconn].savedStates = all_conn_steps[ilayer,iconn]

    def odeRK4(self, func, time, dt, state, params):

        t_half_dt = tf.add(time,dt/2)

        k1 = func(t, state, params)
        k2 = func(t_half_dt, tf.add(state, tf.scalar_mul(dt,k1/2)), params)
        k3 = func(t_half_dt, tf.add(state, tf.scalar_mul(dt,k2/2)), params)
        k4 = func(tf.add(t, d), tf.add(state, tf.scalar_mul(dt,k3)), params)

        state = state + tf.scalar_mul(dt, tf.add_n([k1, 2*k2, 2*k3, k4])/6)

        return state

