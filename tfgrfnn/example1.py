import numpy as np
import tensorflow as tf
import tfgrfnn as tg
from oscillator_types import canonical_hopf
import time as t

dur = 10
dt = 0.01
time = tf.expand_dims(tf.range(10, delta=dt, dtype=tf.float64), axis=0)

s1 = tg.stimulus(name='s1', values=tf.complex(tf.math.cos(2*np.pi*time), tf.math.cos(2*np.pi*time)), fs=1/dt)

l1 = tg.oscillators(name='l1', osctype=canonical_hopf(alpha=1.0, beta1=-1.0, beta2=0.0, epsilon=0.0), 
                    freqspacing='log', freqlims=(0.5, 2.0), nosc=2, savesteps=True)

l1 = tg.connect(source=s1, target=l1, learntype='1freq', learnparams={'lambda_':0, 'mu1':-1, 'mu2':-2, 'epsilon':1, 'kappa':1})

GrFNN = tg.Model(name='GrFNN', layers=[l1], stimuli=[s1], time=tf.squeeze(time))

tic = t.time()
GrFNN = GrFNN.integrate()
toc = t.time() - tic
print(toc)
print(np.abs(GrFNN.layers[0].allsteps))


