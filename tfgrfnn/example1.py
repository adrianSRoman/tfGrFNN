import numpy as np
import tensorflow as tf
import tfgrfnn as tg
import time as t
import matplotlib.pyplot as plt

dur = 100
dt = 0.025
time = tf.range(dur, delta=dt, dtype=tf.float32)
nosc = 201

l1 = tg.neurons(name='l1', 
                osctype='grfnn', 
                params={'alpha':tf.constant(0.01, dtype=tf.float32), 
                        'beta1':tf.constant(-1.0, dtype=tf.float32),
                        'beta2':tf.constant(0.0, dtype=tf.float32),
                        'epsilon':tf.constant(0.0, dtype=tf.float32)}, 
                freqs=tf.constant(np.logspace(np.log10(0.5),np.log(2.0),nosc),dtype=tf.float32), 
                initconds=tf.constant(0.0+1j*0.1, dtype=tf.complex64, shape=(nosc,)))

s1 = tg.stimulus(name='s1', values=tf.expand_dims(tf.expand_dims(0.25*tf.complex(tf.math.cos(2*np.pi*time), tf.math.sin(2*np.pi*time)),0),2), fs=int(1/dt))

l1 = tg.connect(source=s1, target=l1, matrixinit=tf.constant(1.0+1j*1.0, dtype=tf.complex64, shape=(1,1)))

GrFNN = tg.Model(name='GrFNN', layers=[l1], stim=s1)

tic = t.time()
GrFNN = GrFNN.integrate()
toc = t.time() - tic
print(toc)

plt.semilogx(GrFNN.layers[0].freqs,np.abs(GrFNN.layers[0].allsteps[-1]))
plt.ylim([0, 1.3*np.max(np.abs(GrFNN.layers[0].allsteps[-1]))])
plt.grid()
plt.xlabel('Oscillator natural frequency (Hz)')
plt.ylabel('Manitude')
plt.savefig('ex1.png')
