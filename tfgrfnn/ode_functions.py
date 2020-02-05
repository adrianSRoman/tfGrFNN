import tensorflow as tf
import numpy as np
import tfgrfnn as tg

def xdot_ydot(t, tidx, xt_yt, connections, alpha=None, beta1=None, beta2=None, epsilon=None, freqs= None):

    omega = tf.constant(2*np.pi, dtype=tf.float64)

    xt, yt = tf.split(xt_yt, 2, axis=0)
    minusyt_plusxt = tf.concat([tf.scalar_mul(-1,yt), xt], axis=0)

    x2tplusy2t = tf.add(tf.pow(xt, 2),
                        tf.pow(yt, 2))
    x2tplusy2t_x2tplusy2t = tf.concat([x2tplusy2t, 
                                x2tplusy2t], axis=0)
    x2tplusy2tsquared_x2tplusy2tsquared = tf.pow(x2tplusy2t_x2tplusy2t, 2)

    xtnew_ytnew = tf.add_n([
                            tf.scalar_mul(alpha, xt_yt),
                            tf.scalar_mul(omega, minusyt_plusxt),
                            tf.scalar_mul(beta1, tf.multiply(xt_yt, x2tplusy2t_x2tplusy2t)),
                            tf.divide(
                                tf.scalar_mul(tf.multiply(epsilon, beta2), 
                                    tf.multiply(xt_yt, x2tplusy2tsquared_x2tplusy2tsquared)),
                                tf.subtract(tf.constant(1.0, dtype=tf.float64), 
                                    tf.scalar_mul(epsilon, x2tplusy2t_x2tplusy2t)))
                            ])

    csrt_csit = tf.add_n([compute_input(tidx, conn.source.currstate, 
                                            conn.matrix,
                                            conn.learnparams['learntype']) for conn in connections])
            
    dxdt_dydt = tf.multiply(freqs, tf.add(xtnew_ytnew, csrt_csit))
    
    return dxdt_dydt


def compute_input(time_index, conn_source_state, conn_matrix, conn_learntype):

    def compute_input_1freq(srt_sit=conn_source_state, crt_cit=conn_matrix):

        srt_sit = tf.expand_dims(srt_sit, -1)
        srt, sit = tf.split(srt_sit, 2, axis=0)
        crt, cit = tf.split(crt_cit, 2, axis=0)
        csrt = tf.matmul(crt, srt)
        csit = tf.matmul(cit, sit)

        csrt_csit = tf.squeeze(tf.concat([csrt, csit], axis=0))

        return csrt_csit 

    def compute_input_null(conn=conn):

        return tf.constant(0, dtype=tf.float64, shape=(conn.target.nosc,))

    csrt_csit = tf.case({tf.equal(conn_learntype, 'null'): compute_input_null,
                        tf.equal(conn_learntype, '1freq'): compute_input_1freq})

    return csrt_csit

def crdot_cidot(t, t_idx, xst_yst, crt_cit, xtt_ytt, learnparams):

    learntype = learnparams['learntype']

    dcrt_dcit = tf.case({tf.equal(learntype, 'null'): learn_null,
                        tf.equal(learntype, '1freq'): learn_1freq})

    def learn_null(crt_cit):
        return crt_cit

    def learn_1freq(t, t_idx, xst_yst, crt_cit, xtt_ytt, params):
        
        lambda_ = learnparams['lambda_']
        mu1 = learnparams['mu1']
        mu2 = learnparams['mu2']
        kappa = learnparams['kappa']
        epsilon = learnparams['epsilon']
        freqss = learnparams['freqss']
        freqst = learnparams['freqst']

        xst, yst = tf.split(xst_yst, 2, axis=0)
        crt, cit = tf.split(crt_cit, 2, axis=0)
        xtt, ytt = tf.split(xtt_ytt, 2, axis=0)

        cr2tplusci2t = tf.add(tf.pow(crt, 2),
                            tf.pow(cit, 2))
        cr2tplusci2t_cr2tplusci2t = tf.concat([cr2tplusci2t, 
                                    cr2tplusci2t], axis=0)
        cr2tplusci2tsquared_cr2tplusci2tsquared = tf.concat([cr2tplusci2t_cr2tplusci2t], axis=0)
        xst_xst = tf.concat([xst, xst], axis=0)
        plusytt_minusxtt = tf.concat([ytt, tf.scalar_mul(-1, xtt)], axis=0) 
        yst_yst = tf.concat([yss, yss], axis=0)

        dcrdt_dcidt = tf.add_n(
                            tf.scalar_mul(lambda_, crt_cit),
                            tf.scalar_mul(mu1, tf.multiply(crt_cit,
                                                    cr2tplusci2t_cr2tplusci2t)),
                            tf.divide(    
                                tf.scalar_mul(tf.multiply(epsilonc, mu2), 
                                            tf.multiply(crt_cit,
                                                    cr2tplusci2tsquared_cr2tplusci2tsquared)),
                                tf.subtract(tf.constant(1.0, dtype=tf.float64), 
                                    tf.scalar_mul(epsilonc, tf.multiply(crt_cit,
                                                    cr2tplusci2t_cr2tplusci2t)))),
                            tf.scalar_mul(kappa, tf.add(tf.multiply(xtt_ytt, 
                                                                    xst_xst),
                                                        tf.multiply(plusytt_minusxtt,
                                                                    yst_yst))))

        dcrdt_dcidt = tf.multiply(tf.divide(tf.add(freqss, freqst), 2), dcrdt_dcidt)

        return dcdt_dcit

    return dcrdt_dcidt
