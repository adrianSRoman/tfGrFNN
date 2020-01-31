import tfgrfnn as tg
from oscillator_types import canonical_hopf

l1 = tg.oscillators(name='l1', osctype=canonical_hopf(alpha=1.0, beta1=-1.0, beta2=0.0, epsilon=0.0), 
                    freqspacing='log', freqlims=(0.5, 2), nosc=2, savesteps=True)
l2 = tg.oscillators(name='l1', osctype=canonical_hopf(alpha=1.0, beta1=-1.0, beta2=0.0, epsilon=0.0), 
                    freqspacing='log', freqlims=(0.5, 2), nosc=2, savesteps=True)

GrFNN = tg.Model(name='GrFNN', layers=[l1])

GrFNN.integrate()

