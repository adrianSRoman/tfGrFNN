import numpy as np
import tensorflow as tf
import tfgrfnn as tg
from oscillator_types import canonical_hopf
import time as t
import matplotlib.pyplot as plt

dur = 50
dt = 0.025
time = tf.range(dur, delta=dt, dtype=tf.float64)
nosc = 201

l1 = tg.oscillators(name='l1', osctype=canonical_hopf(alpha=0.0, beta1=-1.0, beta2=-1.0, epsilon=1.0), 
                    freqspacing='log', freqlims=(0.5, 2.0), nosc=nosc, initconds=tf.constant(0.0+1j*0.3, dtype=tf.complex128, shape=(nosc,)))

s1 = tg.stimulus(name='s1', values=0.25*tf.complex(tf.math.cos(2*np.pi*time), tf.math.sin(2*np.pi*time)), fs=1/dt)

l1 = tg.connect(source=s1, target=l1)

GrFNN = tg.Model(name='GrFNN', layers=[l1], stim=s1, time=tf.squeeze(time))

tic = t.time()
GrFNN = GrFNN.integrate()
toc = t.time() - tic
print(toc)

plt.semilogx(GrFNN.layers[0].freqs,np.abs(GrFNN.layers[0].allsteps[-1]))
plt.ylim([0 1.3*np.max(np.abs(GrFNN.layers[0].allsteps[-1]))])
plt.grid()
plt.xlabel('Oscillator natural frequency (Hz)')
plt.ylabel('Manitude')
plt.savefig('test.png')
