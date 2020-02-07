import numpy as np
import tensorflow as tf
import tfgrfnn as tg
from oscillator_types import canonical_hopf
import time as t

dur = 10
dt = 0.01
time = tf.range(dur, delta=dt, dtype=tf.float64)

s1 = tg.stimulus(name='s1', values=tf.complex(tf.math.cos(2*np.pi*time), tf.math.cos(2*np.pi*time)), fs=1/dt)

l1 = tg.oscillators(name='l1', osctype=canonical_hopf(alpha=1.0, beta1=-1.0, beta2=0.0, epsilon=0.0), 
                    freqspacing='log', freqlims=(0.5, 3.0), nosc=256)

l1 = tg.connect(source=s1, target=l1, learnparams={'learntype':'1freq', 
                                                    'lambda_':tf.constant(0, dtype=tf.float64), 
                                                    'mu1':tf.constant(-1, dtype=tf.float64), 
                                                    'mu2':tf.constant(-2, dtype=tf.float64), 
                                                    'epsilon':tf.constant(1, dtype=tf.float64), 
                                                    'kappa':tf.constant(1, dtype=tf.float64)})

GrFNN = tg.Model(name='GrFNN', layers=[l1], stim=s1, time=tf.squeeze(time))

tic = t.time()
GrFNN = GrFNN.integrate()
toc = t.time() - tic
print(toc)
tf.print(GrFNN.layers[0].allsteps)
tf.print(tf.shape(GrFNN.layers[0].allsteps))
