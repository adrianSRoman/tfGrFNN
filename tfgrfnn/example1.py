import numpy as np
import tensorflow as tf
import tfgrfnn as tg
import matplotlib.pyplot as plt


# simulation duration values
dur = 50
dt = 0.025
fs = int(1/dt)
time = tf.range(dur, delta=dt, dtype=tf.float16)
# GrFNN network values and parameters
N = 201
freqs = tf.constant(np.logspace(np.log10(0.5), np.log10(2.0),N), dtype=tf.float16)
initconds = tf.constant(0.01+1j*0.0, dtype=tf.complex32, shape=(N,))
params_dict = {'alpha':tf.constant(0.0, dtype=tf.float16), 
             'beta1':tf.constant(-1.0, dtype=tf.float16),
             'beta2':tf.constant(-1.0, dtype=tf.float16),
             'epsilon':tf.constant(1.0, dtype=tf.float16)}
# stimulus values
amplitude = tf.constant(0.25+1j*0.0, dtype=tf.complex32)
freq = tf.constant(1.0, dtype=tf.float16)
theta = 2*np.pi*time*freq
stim_values = tf.multiply(amplitude, 
                        tf.reshape(tf.complex(tf.math.cos(theta), 
                                                tf.math.sin(theta)),
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
# connect stimulus and oscillators
l1 = tg.connect(source=s1, target=l1, 
                matrixinit=tf.constant(1.0+1j*0.0, 
                                        dtype=tf.complex32, shape=(1,N)))
# define the model
GrFNN = tg.Model(name='GrFNN', layers=[l1], stim=s1)
# integrate 
GrFNN = GrFNN.integrate()


# plot results
nxticks = 6
nyticks = 10
plt.subplot(1,2,1)
plt.semilogx(GrFNN.layers[0].params['freqs'],np.abs(GrFNN.layers[0].states[0,:,-1]))
plt.ylim([0, 1.3*np.max(np.abs(GrFNN.layers[0].states[0,:,-1]))])
plt.grid()
plt.xlabel('Oscillator natural frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude of oscillators (t = '+str(dur)+'s)')
ax = plt.subplot(1,2,2)
plt.imshow(np.abs(GrFNN.layers[0].states[0]), aspect = 'auto')
ax.set_xticklabels([str(round(float(label),2)) for label in GrFNN.time.numpy()\
        [np.round(np.linspace(0,len(GrFNN.time)-1,nxticks)).astype(int)]])
ax.set_yticklabels([str(round(float(label),2)) for label in GrFNN.layers[0].params['freqs'].numpy()[::-1]\
        [np.round(np.linspace(0,len(GrFNN.layers[0].params['freqs'])-1,nyticks)).astype(int)]])
plt.xlabel('Time (s)')
plt.ylabel('Oscillator natural frequency (Hz)')
plt.colorbar().set_label('Amplitude')
plt.title('Amplitude of oscillators over time')
plt.tight_layout()
plt.savefig('example1.png')
