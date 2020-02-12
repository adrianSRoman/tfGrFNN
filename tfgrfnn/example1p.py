import numpy as np
import tensorflow as tf
import tfgrfnn as tg
from oscillator_types import canonical_hopf
import time as t
import matplotlib.pyplot as plt

dur = 100
dt = 0.025
time = tf.range(dur, delta=dt, dtype=tf.float64)
nosc = 201

l1 = tg.oscillators(name='l1', osctype=canonical_hopf(alpha=1.0, beta1=-1.0, beta2=-1000.0, epsilon=1.0), 
                    freqspacing='log', freqlims=(0.5, 2.0), nosc=nosc, initconds=tf.constant(0.01+1j*0.01, dtype=tf.complex128, shape=(nosc,)))

l1 = tg.connect(source=l1, target=l1, matrixinit=1, learnparams={'learntype':'1freq',
                                                                'lambda_':tf.constant(0.001, dtype=tf.float64),
                                                                'mu1':tf.constant(-1.0, dtype=tf.float64),
                                                                'mu2':tf.constant(-50.0, dtype=tf.float64),
                                                                'epsilon':tf.constant(16.0, dtype=tf.float64),
                                                                'kappa':tf.constant(1.0, dtype=tf.float64)})

GrFNN = tg.Model(name='GrFNN', layers=[l1], time=tf.squeeze(time))

tic = t.time()
GrFNN = GrFNN.integrate()
toc = t.time() - tic
print(toc)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.semilogx(GrFNN.layers[0].freqs,np.abs(GrFNN.layers[0].allsteps[-1]))
ax1.grid()
ax1.set_ylim([0, 1.3*np.max(np.abs(GrFNN.layers[0].allsteps[-1]))])
ax1.set_xlabel('Oscillator natural frequency (Hz)')
ax1.set_ylabel('Magnitude')
ax2.imshow(np.abs(GrFNN.layers[0].connections[0].allmatrixsteps[-1]))
ticks = np.linspace(0, GrFNN.layers[0].nosc-1, 5, dtype=int)
ticklabels = ["{:.1f}".format(i) for i in GrFNN.layers[0].freqs[ticks]]
ax2.set_xticks(ticks)
ax2.set_xticklabels(ticklabels)
ax2.set_yticks(ticks)
ax2.set_yticklabels(ticklabels)
ax2.set_xlabel('Oscillator natural frequency (Hz)')
ax2.set_ylabel('Oscillator natural frequency (Hz)')
plt.savefig('test.png')
