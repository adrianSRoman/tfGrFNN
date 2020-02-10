import numpy as np
import tensorflow as tf
import tfgrfnn as tg
from oscillator_types import canonical_hopf
import time as t
import matplotlib.pyplot as plt

dur = 50
dt = 0.01
time = tf.range(dur, delta=dt, dtype=tf.float64)
nosc = 201

l1 = tg.oscillators(name='l1', osctype=canonical_hopf(alpha=0.1, beta1=-1.0, beta2=-1000.0, epsilon=1.0), 
                    freqspacing='log', freqlims=(0.5, 2.0), nosc=nosc, initconds=tf.constant(0.0+1j*0.3, dtype=tf.complex128, shape=(nosc,)))

s1 = tg.stimulus('s1', values=tf.constant(0, dtype=tf.complex128, shape=tf.shape(time)), fs=1/dt)

l1 = tg.connect(source=s1, target=l1)

l1 = tg.connect(source=l1, target=l1, matrixinit=1, learnparams={'learntype':'1freq',
                                                                'lambda_':tf.constant(0.001, dtype=tf.float64),
                                                                'mu1':tf.constant(-1.0, dtype=tf.float64),
                                                                'mu2':tf.constant(-50.0, dtype=tf.float64),
                                                                'epsilon':tf.constant(16.0, dtype=tf.float64),
                                                                'kappa':tf.constant(1.0, dtype=tf.float64)})

GrFNN = tg.Model(name='GrFNN', layers=[l1], stim=s1, time=tf.squeeze(time))

tic = t.time()
GrFNN = GrFNN.integrate()
toc = t.time() - tic
print(toc)

plt.subplot(1,2,1)
plt.semilogx(GrFNN.layers[0].freqs,np.abs(GrFNN.layers[0].allsteps[-1]))
plt.subplot(1,2,2)
plt.imshow(np.abs(GrFNN.layers[0].connections[1].allmatrixsteps[-1]))
plt.savefig('test.png')
