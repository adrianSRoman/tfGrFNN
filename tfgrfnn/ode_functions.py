import tensorflow as tf
import numpy as np
import tfgrfnn as tg

def xdot_ydot(t, xt_yt, connmats_state, connections, sources_state, alpha=None, beta1=None, beta2=None, epsilon=None, freqs= None):

    omega = tf.constant(2*np.pi, dtype=tf.float32)

    xt, yt = tf.split(xt_yt, 2, axis=0)
    minusyt_plusxt = tf.concat([tf.scalar_mul(-1,yt), xt], axis=0)

    x2tplusy2t = tf.add(tf.pow(xt, 2),
                        tf.pow(yt, 2))
    x2tplusy2t_x2tplusy2t = tf.concat([x2tplusy2t, 
                                x2tplusy2t], axis=0)
    x2tplusy2tsquared_x2tplusy2tsquared = tf.pow(x2tplusy2t_x2tplusy2t, 2)

    xtnew_ytnew = tf.add_n([tf.scalar_mul(alpha, xt_yt),
                            tf.scalar_mul(omega, minusyt_plusxt),
                            tf.scalar_mul(beta1, tf.multiply(xt_yt, x2tplusy2t_x2tplusy2t)),
                            tf.divide(
                                tf.scalar_mul(tf.multiply(epsilon, beta2), 
                                    tf.multiply(xt_yt, x2tplusy2tsquared_x2tplusy2tsquared)),
                                tf.subtract(tf.constant(1.0, dtype=tf.float32), 
                                    tf.scalar_mul(epsilon, x2tplusy2t_x2tplusy2t)))])

    csrt_csit = tf.add_n([tf.scalar_mul(connections[iconn].learnparams['weight'], compute_input(connmat_state, 
                            sources_state[connections[iconn].sourceintid], 
                            connections[iconn].learnparams['learntypeint']))
                        for iconn, connmat_state in enumerate(connmats_state)]) if connmats_state else 0
            
    dxdt_dydt = tf.multiply(freqs, tf.add(xtnew_ytnew, csrt_csit))
    
    return dxdt_dydt


def compute_input(connmat_state, source_state, learntypeint):

    def compute_input_nolearning(srt_sit=source_state, crt_cit=connmat_state):

        srt, sit = tf.split(srt_sit, 2, axis=0)
        crt, cit = tf.split(crt_cit, 2, axis=0)
        csrt = tf.matmul(crt, srt)
        csit = tf.matmul(cit, sit)

        csrt_csit = tf.squeeze(tf.concat([csrt, csit], axis=0))

        return csrt_csit 

    def compute_input_1freq(srt_sit=source_state, crt_cit=connmat_state):

        srt, sit = tf.split(srt_sit, 2, axis=0)
        crt, cit = tf.split(crt_cit, 2, axis=0)
        csrt = tf.matmul(crt, srt)
        csit = tf.matmul(cit, sit)

        csrt_csit = tf.squeeze(tf.concat([csrt, csit], axis=0))

        return csrt_csit 

    csrt_csit = tf.switch_case(learntypeint,
                                branch_fns={0: compute_input_nolearning,
                                            1: compute_input_1freq})

    return csrt_csit

def crdot_cidot(t, xst_yst, crt_cit, xtt_ytt, learnparams):

    def nolearning(t=t, xst_yst=xst_yst, crt_cit=crt_cit, xtt_ytt=xtt_ytt, learnparams=learnparams):
        
        return tf.constant(0, dtype=tf.float32, shape=crt_cit.shape)

    def learn_1freq(t=t, xst_yst=xst_yst, crt_cit=crt_cit, xtt_ytt=xtt_ytt, learnparams=learnparams):
        
        lambda_ = learnparams['lambda_']
        mu1 = learnparams['mu1']
        mu2 = learnparams['mu2']
        kappa = learnparams['kappa']
        epsilon = learnparams['epsilon']
        freqss = tf.expand_dims(learnparams['freqss'], 0)
        freqst = tf.expand_dims(learnparams['freqst'], -1)

        xst, yst = tf.split(xst_yst, 2, axis=0)
        crt, cit = tf.split(crt_cit, 2, axis=0)
        xtt, ytt = tf.split(xtt_ytt, 2, axis=0)

        cr2tplusci2t = tf.add(tf.pow(crt, 2),
                            tf.pow(cit, 2))
        cr2tplusci2t_cr2tplusci2t = tf.concat([cr2tplusci2t, 
                                    cr2tplusci2t], axis=0)
        cr2tplusci2tsquared_cr2tplusci2tsquared = tf.pow(cr2tplusci2t_cr2tplusci2t, 2)

        dcrdt_dcidt = tf.add_n([tf.scalar_mul(lambda_, crt_cit),
                            tf.scalar_mul(mu1, tf.multiply(crt_cit, cr2tplusci2t_cr2tplusci2t)),
                            tf.divide(tf.scalar_mul(tf.multiply(epsilon, mu2), 
                                        tf.multiply(crt_cit, cr2tplusci2tsquared_cr2tplusci2tsquared)),
                                    tf.subtract(tf.constant(1.0, dtype=tf.float32), 
                                        tf.scalar_mul(epsilon, cr2tplusci2t_cr2tplusci2t))),
                            tf.scalar_mul(kappa, tf.concat([tf.add(tf.matmul(tf.expand_dims(xtt,-1), 
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

    learntype = learnparams['learntypeint']

    dcrdt_dcidt = tf.switch_case(learntype,
                                    branch_fns={0: nolearning,
                                                1: learn_1freq})

    return dcrdt_dcidt
