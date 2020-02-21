import numpy as np
import tensorflow as tf
import tfgrfnn as tg
from oscillator_types import canonical_hopf
import time as t
import matplotlib.pyplot as plt

dur = 10
dt = 0.05
time = tf.range(dur, delta=dt, dtype=tf.float64)
nosc = 2


#canonical_hopf_params = {'alpha': tf.Variable(0.0, trainable=True, dtype=tf.float64),
#                        'beta1': tf.Variable(0.0, trainable=True, dtype=tf.float64),
#                        'beta2': tf.Variable(0.0, trainable=True, dtype=tf.float64),
#                        'epsilon': tf.Variable(1.0, trainable=True, dtype=tf.float64)}

canonical_hopf_params = {'alpha': tf.Variable(tf.random.normal((), stddev=0.00001, dtype=tf.float64), trainable=True, dtype=tf.float64),
                        'beta1': tf.Variable(tf.random.normal((), stddev=0.00001, dtype=tf.float64), trainable=True, dtype=tf.float64),
                        'beta2': tf.Variable(tf.random.normal((), stddev=0.00001,  dtype=tf.float64), trainable=True, dtype=tf.float64),
                        'epsilon': tf.Variable(tf.random.normal((), stddev=0.00001, dtype=tf.float64), trainable=True, dtype=tf.float64)}

l1 = tg.oscillators(name='l1', osctype=canonical_hopf_params, 
                    freqspacing='log', freqlims=(1.0, 2.0), nosc=nosc, 
                    #initconds=tf.complex(tf.constant(1.0, dtype=tf.float64, shape=(nosc,)), 
                    #                    tf.constant(0.0, dtype=tf.float64, shape=(nosc,))))
                    initconds=tf.complex(tf.random.truncated_normal((nosc,), stddev=1.0,  dtype=tf.float64),
                                            tf.random.truncated_normal((nosc,), dtype=tf.float64, stddev=1.0)))

s1 = tg.stimulus(name='s1', values=0.25*tf.complex(0*time, 0*time), fs=1/dt)
target = tf.constant(1, dtype=tf.float64, shape=(3,1))

l1 = tg.connect(source=s1, target=l1, matrixinit=0.0+1j*0.0)

GrFNN = tg.Model(name='GrFNN', layers=[l1], stim=s1, time=time)

def train_step(GrFNN, target):
    with tf.GradientTape() as tape:
        GrFNN = GrFNN.integrate()
        mse = tf.losses.MeanSquaredError()
        curr_loss = mse(target,
                        tf.abs(GrFNN.layers[0].allsteps[-3:,0]))
    grads = tape.gradient(curr_loss, [GrFNN.layers[0].params['alpha'], GrFNN.layers[0].params['beta1'],
                                    GrFNN.layers[0].params['beta2'], GrFNN.layers[0].params['epsilon']])
    tf.optimizers.SGD(0.01).apply_gradients(zip(grads, [GrFNN.layers[0].params['alpha'], 
                                                        GrFNN.layers[0].params['beta1'],
                                                        GrFNN.layers[0].params['beta2'], 
                                                        GrFNN.layers[0].params['epsilon']]))

    tf.print('MSE Loss :', tf.reduce_mean(curr_loss))
    tf.print('alpha:', GrFNN.layers[0].params['alpha'])
    tf.print('beta1:', GrFNN.layers[0].params['beta1'])
    tf.print('beta2:', GrFNN.layers[0].params['beta2'])
    tf.print('epsilon:', GrFNN.layers[0].params['epsilon'])

num_epochs = 200

for e in range(num_epochs):
    train_step(GrFNN, target)

    if e % 10 == 0:
        plt.plot(time[:-1],np.real(GrFNN.layers[0].allsteps[:,0]))
        plt.plot(time[:-1],np.imag(GrFNN.layers[0].allsteps[:,0]))
        plt.plot(time[:-1],np.abs(GrFNN.layers[0].allsteps[:,0]))
        plt.grid()
        plt.savefig('gd_step'+str(e)+'.png')
        plt.clf()
