import numpy as np
import tensorflow as tf
import tfgrfnn as tg
from oscillator_types import canonical_hopf
import time as t
import matplotlib.pyplot as plt
from matplotlib import gridspec

dur = 10
dt = 0.05
time = tf.range(dur, delta=dt, dtype=tf.float64)
nosc = 2

canonical_hopf_params = {'alpha': tf.Variable(tf.random.uniform((), minval=-0.00001, maxval=0.00001, dtype=tf.float64), trainable=True, dtype=tf.float64),
                        'beta1': tf.Variable(tf.random.uniform((), minval=-0.00001, maxval=0.00001, dtype=tf.float64), trainable=True, dtype=tf.float64),
                        'beta2': tf.Variable(tf.random.uniform((), minval=-0.00001, maxval=0, dtype=tf.float64), trainable=True, dtype=tf.float64),
                        'epsilon': tf.Variable(tf.random.uniform((), minval=0, maxval=0.00001, dtype=tf.float64), trainable=True, dtype=tf.float64)}

l1 = tg.oscillators(name='l1', osctype=canonical_hopf_params, 
                    freqspacing='log', freqlims=(1.0, 2.0), nosc=nosc, 
                    #initconds=tf.complex(tf.constant(1.0, dtype=tf.float64, shape=(nosc,)), 
                    #                    tf.constant(0.0, dtype=tf.float64, shape=(nosc,))))
                    initconds=tf.complex(tf.random.truncated_normal((nosc,), stddev=1.0,  dtype=tf.float64),
                                            tf.random.truncated_normal((nosc,), dtype=tf.float64, stddev=1.0)))

s1 = tg.stimulus(name='s1', values=tf.complex(0*time, 0*time), fs=1/dt) # no stimulus
target = tf.constant(1, dtype=tf.float64, shape=(6,1))

l1 = tg.connect(source=s1, target=l1, matrixinit=0.0+1j*0.0)

GrFNN = tg.Model(name='GrFNN', layers=[l1], stim=s1, time=time)

def train_step(GrFNN, target):
    with tf.GradientTape() as tape:
        GrFNN = GrFNN.integrate()
        mse = tf.losses.MeanSquaredError()
        curr_loss = mse(target,
                        tf.abs(GrFNN.layers[0].allsteps[-6:,0]))
    grads = tape.gradient(curr_loss, [GrFNN.layers[0].params['alpha'], GrFNN.layers[0].params['beta1'],
                                    GrFNN.layers[0].params['beta2'], GrFNN.layers[0].params['epsilon']])
    tf.optimizers.SGD(0.008).apply_gradients(zip(grads, [GrFNN.layers[0].params['alpha'], 
                                                        GrFNN.layers[0].params['beta1'],
                                                        GrFNN.layers[0].params['beta2'], 
                                                        GrFNN.layers[0].params['epsilon']]))

    tf.print('Osc mag:', tf.abs(GrFNN.layers[0].allsteps[-1,0]))
    tf.print('MSE Loss :', tf.reduce_mean(curr_loss))
    tf.print('alpha:', GrFNN.layers[0].params['alpha'])
    tf.print('beta1:', GrFNN.layers[0].params['beta1'])
    tf.print('beta2:', GrFNN.layers[0].params['beta2'])
    tf.print('epsilon:', GrFNN.layers[0].params['epsilon'])

    return GrFNN

num_epochs = 100

fig = plt.figure(figsize=(13,7))
gs = gridspec.GridSpec(2,4)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[0,3])
ax4 = fig.add_subplot(gs[1,:])
param_iters = np.zeros((num_epochs+1,4))
param_iters[0,0]=GrFNN.layers[0].params['alpha'].numpy()
param_iters[0,1]=GrFNN.layers[0].params['beta1'].numpy()
param_iters[0,2]=GrFNN.layers[0].params['beta2'].numpy()
param_iters[0,3]=GrFNN.layers[0].params['epsilon'].numpy()
for e in range(num_epochs):
    GrFNN = train_step(GrFNN, target)

    param_iters[e+1,0]=GrFNN.layers[0].params['alpha'].numpy()
    param_iters[e+1,1]=GrFNN.layers[0].params['beta1'].numpy()
    param_iters[e+1,2]=GrFNN.layers[0].params['beta2'].numpy()
    param_iters[e+1,3]=GrFNN.layers[0].params['epsilon'].numpy()

    if e == 0:

        ax0.plot(time[:-1],np.real(GrFNN.layers[0].allsteps[:,0]))
        ax0.plot(time[:-1],np.imag(GrFNN.layers[0].allsteps[:,0]))
        ax0.plot(time[:-1],np.abs(GrFNN.layers[0].allsteps[:,0]))
        ax0.set_xlabel('time (s)')
        ax0.set_ylim([-1.5, 1.5])
        ax0.set_xlim([0, dur])
        ax0.grid()

    if e == num_epochs//3:

        ax1.plot(time[:-1],np.real(GrFNN.layers[0].allsteps[:,0]))
        ax1.plot(time[:-1],np.imag(GrFNN.layers[0].allsteps[:,0]))
        ax1.plot(time[:-1],np.abs(GrFNN.layers[0].allsteps[:,0]))
        ax1.grid()
        ax1.set_ylim([-1.5, 1.5])
        ax1.set_xlim([0, dur])
        ax1.set_xlabel('time (s)')

    if e == 2*(num_epochs//3):

        ax2.plot(time[:-1],np.real(GrFNN.layers[0].allsteps[:,0]))
        ax2.plot(time[:-1],np.imag(GrFNN.layers[0].allsteps[:,0]))
        ax2.plot(time[:-1],np.abs(GrFNN.layers[0].allsteps[:,0]))
        ax2.grid()
        ax2.set_ylim([-1.5, 1.5])
        ax2.set_xlim([0, dur])
        ax2.set_xlabel('time (s)')

    if e == 3*(num_epochs//3):

        ax3.plot(time[:-1],np.real(GrFNN.layers[0].allsteps[:,0]), label="Real")
        ax3.plot(time[:-1],np.imag(GrFNN.layers[0].allsteps[:,0]), label="Imag")
        ax3.plot(time[:-1],np.abs(GrFNN.layers[0].allsteps[:,0]), label=r"|z|")
        ax3.grid()
        ax3.set_ylim([-1.5, 1.5])
        ax3.set_xlim([0, dur])
        ax3.set_xlabel('time (s)')
        ax3.legend(loc="lower right")

ax4.plot(param_iters)
ax4.grid()
ax4.legend([r'$\alpha$', r'$\beta_1$', r'$\beta_2$', r'$\epsilon$'], loc="lower right")
ax4.set_xlabel('Gradient Descent steps')
ax4.set_xlim([0, num_epochs])
fig.savefig('example1gd.png')

