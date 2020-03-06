import numpy as np
import tensorflow as tf
import tfgrfnn as tg
import matplotlib.pyplot as plt

# simulation duration values
dur = 100
dt = 0.025
fs = int(1/dt)
time = tf.range(dur, delta=dt, dtype=tf.float32)

# GrFNN network values and parameters
N = 201
freqs = tf.constant(np.logspace(np.log10(0.5),np.log10(2.0),N),dtype=tf.float32)
initconds = tf.constant(0.0+1j*0.1, dtype=tf.complex64, shape=(N,))
params_dict = {'alpha':tf.constant(0.01, dtype=tf.float32), 
             'beta1':tf.constant(-1.0, dtype=tf.float32),
             'beta2':tf.constant(0.0, dtype=tf.float32),
             'epsilon':tf.constant(0.0, dtype=tf.float32)}

# stimulus values
amplitude = tf.constant(0.25+0.0*1j, dtype=tf.complex64)
freq = tf.constant(1.0, dytpe=tf.float32)
stim_values = tf.scalar_mul(amplitude, 
                        tf.reshape(tf.complex(tf.math.cos(2*np.pi*time), 
                                                tf.math.sin(2*np.pi*time)),
                                    [1,len(time),1]))

# define the layer of oscillators
l1 = tg.neurons(name = 'l1', 
                osctype = 'grfnn', 
                params = params_dict,
                freqs = freqs, 
                initconds= initconds)

# define the stimulus 
s1 = tg.stimulus(name = 's1', 
                values = stim_values,
                fs = fs)

l1 = tg.connect(source=s1, target=l1, matrixinit=tf.constant(1.0+1j*1.0, dtype=tf.complex64, shape=(1,N)))

GrFNN = tg.Model(name='GrFNN', layers=[l1], stim=s1)

GrFNN = GrFNN.integrate()

plt.semilogx(GrFNN.layers[0].params['freqs'],np.abs(GrFNN.layers[0].states[0,:,-1]))
plt.ylim([0, 1.3*np.max(np.abs(GrFNN.layers[0].states[0,:,-1]))])
plt.grid()
plt.xlabel('Oscillator natural frequency (Hz)')
plt.ylabel('Manitude')
plt.savefig('ex1.png')
