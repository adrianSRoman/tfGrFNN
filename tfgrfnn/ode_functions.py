import tensorflow as tf
import numpy as np

def xdot_ydot(t, x_y, alpha, beta1, beta2, epsilon, freqs,
                connmats_state, sources_state, conns_source_intid, 
                conns_typeint, conns_weight, dtype=tf.float32):

    omega = tf.constant(2*np.pi, dtype=dtype)

    x, y = tf.split(x_y, 2, axis=1)

    x2plusy2 = tf.add(tf.pow(x, 2),
                        tf.pow(y, 2))
    x2plusy2squared = tf.pow(x2plusy2, 2)

    xnew = tf.add_n([tf.multiply(alpha, x),
                        tf.multiply(omega, tf.multiply(-1.0, y)),
                        tf.multiply(beta1, tf.multiply(x, x2plusy2)),
                        tf.divide(
                            tf.multiply(tf.multiply(epsilon,beta2),
                                tf.multiply(x, x2plusy2squared)),
                            tf.subtract(tf.constant(1.0, dtype=dtype),
                                tf.multiply(epsilon, x2plusy2)))])

    ynew = tf.add_n([tf.multiply(alpha, y),
                        tf.multiply(omega, x),
                        tf.multiply(beta1, tf.multiply(y, x2plusy2)),
                        tf.divide(
                            tf.multiply(tf.multiply(epsilon,beta2),
                                tf.multiply(y, x2plusy2squared)),
                            tf.subtract(tf.constant(1.0, dtype=dtype),
                                tf.multiply(epsilon, x2plusy2)))])

    xnew = tf.multiply(freqs, xnew)
    ynew = tf.multiply(freqs, ynew)
    xnew_ynew = tf.concat([xnew, ynew], axis=1)

    csr_csi = tf.add_n([compute_input(connmat_state, sources_state[source_intid],
                            x_y, conntypeint, epsilon, input_weight, freqs, dtype)
                        for connmat_state, source_intid, conntypeint, input_weight \
                            in zip(connmats_state, conns_source_intid, conns_typeint, \
                            conns_weight)]) if connmats_state else 0
            
    dxdt_dydt = tf.add(xnew_ynew, csr_csi)
    
    return dxdt_dydt


def compute_input(connmat_state, source_state, target_state, conntypeint, 
                epsilon, input_weight, freqs, dtype):

    def compute_input_1freq(sr_si=source_state, cr_ci=connmat_state, 
                            input_weight=input_weight, freqs=freqs):

        sr, si = tf.split(sr_si, 2, axis=1)
        cr, ci = tf.split(cr_ci, 2, axis=0)
        csr = tf.matmul(sr, cr) - tf.matmul(si, ci)
        csi = tf.matmul(si, cr) + tf.matmul(sr, ci)
        csr = tf.multiply(input_weight, csr)
        csi = tf.multiply(input_weight, csi)
        csr = tf.multiply(freqs, csr)
        csi = tf.multiply(freqs, csi)

        csr_csi = tf.concat([csr, csi], axis=1)

        return csr_csi 

    def compute_input_allfreq(sr_si=source_state, tr_ti=target_state, 
                                cr_ci=connmat_state, epsilon=epsilon, 
                                input_weight=input_weight, freqs=freqs):

        sr, si = tf.split(sr_si, 2, axis=1)
        tr, ti = tf.split(tr_ti, 2, axis=1)
        cr, ci = tf.split(cr_ci, 2, axis=0)

        sqrteps = tf.sqrt(tf.subtract(epsilon,0.05))
        sr2 = tf.pow(sr, 2)
        si2 = tf.pow(si, 2)
        ti2 = tf.pow(ti, 2)

        one_min_re2 = tf.pow(tf.subtract(tf.constant(1.0, dtype=dtype),
                                    tf.multiply(sr, sqrteps)), 2)
        Pdenominator = tf.add(one_min_re2, tf.multiply(si2, epsilon))
        Pn1r = tf.divide(tf.add_n([tf.multiply(-sr2, sqrteps),
                                    sr,
                                    tf.multiply(-si2, sqrteps)]),
                        Pdenominator)
        Pn1i = tf.divide(si,
                        Pdenominator)
        Pn2r = tf.divide(tf.add(tf.constant(1.0, dtype=dtype),
                                tf.multiply(-sr,sqrteps)),
                        Pdenominator)
        Pn2i = tf.divide(-tf.multiply(si, sqrteps),
                        Pdenominator)

        one_min_re2 = tf.pow(tf.subtract(tf.constant(1.0, dtype=dtype),
                                    tf.multiply(tr, sqrteps)), 2)
        Adenominator = tf.add(one_min_re2, tf.multiply(ti2, epsilon))
        Ar = tf.divide(tf.add(tf.constant(1.0, dtype=dtype),
                                tf.multiply(-tr, sqrteps)),
                        Adenominator)
        Ai = tf.divide(-tf.multiply(ti, sqrteps),
                        Adenominator)

        Pnr = tf.multiply(Pn1r,Pn2r) - tf.multiply(Pn1i,Pn2i) 
        Pni = tf.multiply(Pn1i,Pn2r) + tf.multiply(Pn1r,Pn2i) 

        csr = tf.matmul(Pnr,cr) - tf.matmul(Pni,ci)
        csi = tf.matmul(Pni,cr) + tf.matmul(Pnr,ci)

        csr = tf.multiply(Ar,csr) - tf.multiply(Ai,csi)
        csi = tf.multiply(Ai,csr) + tf.multiply(Ar,csi)

        csr = tf.multiply(input_weight, csr)
        csi = tf.multiply(input_weight, csi)
        csr = tf.multiply(freqs, csr)
        csi = tf.multiply(freqs, csi)

        csr_csi = tf.concat([csr, csi], axis=1)

        return csr_csi 

    csr_csi = tf.switch_case(conntypeint,
                                branch_fns={0: compute_input_1freq,
                                            1: compute_input_allfreq})

    return csr_csi
   
''' 
def crdot_cidot(t, xst_yst, crt_cit, xtt_ytt, params):

    def nolearning(crt_cit=crt_cit):
        
        return tf.constant(0, dtype=tf.float32, shape=crt_cit.shape)

    def learn_1freq(t=t, xst_yst=xst_yst, crt_cit=crt_cit, xtt_ytt=xtt_ytt, params=params):
        
        lambda_ = params['lambda_']
        mu1 = params['mu1']
        mu2 = params['mu2']
        kappa = params['kappa']
        epsilon = params['epsilon']
        freqss = tf.expand_dims(params['freqss'], 0)
        freqst = tf.expand_dims(params['freqst'], -1)

        xst, yst = tf.split(xst_yst, 2, axis=0)
        crt, cit = tf.split(crt_cit, 2, axis=0)
        xtt, ytt = tf.split(xtt_ytt, 2, axis=0)

        cr2tplusci2t = tf.add(tf.pow(crt, 2),
                            tf.pow(cit, 2))
        cr2tplusci2t_cr2tplusci2t = tf.concat([cr2tplusci2t, 
                                    cr2tplusci2t], axis=0)
        cr2tplusci2tsquared_cr2tplusci2tsquared = tf.pow(cr2tplusci2t_cr2tplusci2t, 2)

        dcrdt_dcidt = tf.add_n([tf.multiply(lambda_, crt_cit),
                            tf.multiply(mu1, tf.multiply(crt_cit, cr2tplusci2t_cr2tplusci2t)),
                            tf.divide(tf.multiply(tf.multiply(epsilon, mu2), 
                                        tf.multiply(crt_cit, cr2tplusci2tsquared_cr2tplusci2tsquared)),
                                    tf.subtract(tf.constant(1.0, dtype=tf.float32), 
                                        tf.multiply(epsilon, cr2tplusci2t_cr2tplusci2t))),
                            tf.multiply(kappa, tf.concat([tf.add(tf.matmul(tf.expand_dims(xtt,-1), 
                                                                            tf.expand_dims(xst,0)), 
                                                                tf.matmul(tf.expand_dims(ytt,-1), 
                                                                            tf.expand_dims(yst,0))),
                                                            tf.add(tf.matmul(tf.expand_dims(ytt,-1), 
                                                                            tf.expand_dims(xst,0)),
                                                                tf.matmul(tf.expand_dims(-xtt,-1), 
                                                                            tf.expand_dims(yst,0)))], axis=0))])

        fmat = tf.add(tf.tile(freqst, tf.shape(freqss)), 
                tf.tile(freqss, tf.shape(freqst)))
        dcrdt_dcidt = tf.multiply(tf.divide(tf.concat([fmat, fmat], axis=0), 2), dcrdt_dcidt)

        return dcrdt_dcidt

    if params['learn']:
        dcrdt_dcidt = nolearning(crt_cit) 
    #    learntype = params['typeint']
    #    dcrdt_dcidt = tf.switch_case(learntype,
    #                                    branch_fns={0: learn_1freq,
    #                                                1: learn_allfreq})
    else:
        dcrdt_dcidt = nolearning(crt_cit) 

    return dcrdt_dcidt
'''
