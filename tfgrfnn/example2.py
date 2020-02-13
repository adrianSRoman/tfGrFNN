import numpy as np
import tensorflow as tf
import tfgrfnn as tg
from oscillator_types import canonical_hopf
import time as t
import matplotlib.pyplot as plt

dur = 1.0
dt = 0.00025
time = tf.range(dur, delta=dt, dtype=tf.float64)
nosc = 201

s1 = tg.stimulus(name='s1', values=0.025*tf.complex(tf.math.cos(2*np.pi*100*time), 
                                                    tf.math.sin(2*np.pi*100*time)), fs=1/dt)
l1 = tg.oscillators(name='l1', osctype=canonical_hopf(alpha=0.01, beta1=-1.0, beta2=-10.0, epsilon=1.0), 
                    freqspacing='log', freqlims=(50.0, 200.0), nosc=nosc, 
                    initconds=tf.constant(0.1+1j*0.0, dtype=tf.complex128, shape=(nosc,)))
l1 = tg.connect(source=s1, target=l1, matrixinit=1.0+1j*1.0)

l2 = tg.oscillators(name='l2', osctype=canonical_hopf(alpha=-1.0, beta1=4.0, beta2=-3.0, epsilon=1.0), 
                    freqspacing='log', freqlims=(50.0, 200.0), nosc=nosc, 
                    initconds=tf.constant(0.1+1j*0.0, dtype=tf.complex128, shape=(nosc,)))
l2 = tg.connect(source=l1, target=l2, matrixinit=np.eye(nosc)+1j*np.eye(nosc))

GrFNN = tg.Model(name='GrFNN', layers=[l1,l2], stim=s1, time=time)

tic = t.time()
GrFNN = GrFNN.integrate()
toc = t.time() - tic
print(toc)

plt.subplot(1,2,1)
plt.semilogx(GrFNN.layers[0].freqs,np.abs(GrFNN.layers[0].allsteps[-1]))
plt.ylim([0, 1.3*np.max(np.abs(GrFNN.layers[0].allsteps[-1]))])
plt.grid()
plt.xlabel('Oscillator natural frequency (Hz)')
plt.ylabel('Manitude')
plt.subplot(1,2,2)
plt.semilogx(GrFNN.layers[1].freqs,np.abs(GrFNN.layers[1].allsteps[-1]))
plt.ylim([0, 1.3*np.max(np.abs(GrFNN.layers[1].allsteps[-1]))])
plt.grid()
plt.xlabel('Oscillator natural frequency (Hz)')
plt.ylabel('Manitude')
plt.savefig('ex2.png')
