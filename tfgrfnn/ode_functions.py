import tensorflow as tf
import numpy as np
import tfgrfnn as tg

def xdot_ydot(t, t_idx, xt_yt, connections, alpha=None, beta1=None, beta2=None, epsilon=None, freqs= None, ones=None):

    omega = tf.constant(2*np.pi, dtype=tf.float64)

    xt, yt = tf.split(xt_yt, 2, axis=0)
    minusyt_plusxt = tf.concat([tf.scalar_mul(-1,yt), xt], axis=0)

    x2tplusy2t = tf.add(tf.pow(xt, 2),
                        tf.pow(yt, 2))
    x2tplusy2t_x2tplusy2t = tf.concat([x2tplusy2t, 
                                x2tplusy2t], axis=0)
    x2tplusy2tsquared = tf.pow(x2tplusy2t, 2)
    x2tplusy2tsquared_x2tplusy2tsquared = tf.concat([x2tplusy2tsquared, 
                                x2tplusy2tsquared], axis=0)

    xtnew_ytnew = tf.add_n([
                            tf.scalar_mul(alpha, xt_yt),
                            tf.scalar_mul(omega, minusyt_plusxt),
                            tf.scalar_mul(beta1, tf.multiply(xt_yt, x2tplusy2t_x2tplusy2t)),
                            tf.divide(
                                tf.scalar_mul(tf.multiply(epsilon, beta2), 
                                    tf.multiply(xt_yt, x2tplusy2tsquared_x2tplusy2tsquared)),
                                tf.subtract(ones, 
                                    tf.scalar_mul(epsilon, x2tplusy2t_x2tplusy2t)))
                            ])

    if connections == None:
        dxdt_dydt = tf.multiply(freqs, xtnew_ytnew)
    else:
        csrt_csit = tf.add_n([compute_inputs(t_idx, conn) for conn in connections])
        dxdt_dydt = tf.multiply(freqs, tf.add(xtnew_ytnew, csrt_csit))

    return dxdt_dydt

def compute_inputs(time_index, conn):

    if isinstance(conn.source, tg.oscillators):
        conn_type, srt_sit, crt_cit, params = conn.type, conn.source.state, conn.matrix, conn.params
    else: 
        if isinstance(time_index, int):
            conn_type, srt_sit, crt_cit, params = conn.type, conn.source.state[time_index], conn.matrix, conn.params
        else:
            conn_type, srtm1_sitm1, srtp1_sitp1, crt_cit, params = conn.type, conn.source.state[int(time_index-0.5)], conn.source.state[int(time_index+0.5)], conn.matrix, conn.params
            srt_sit = srtm1_sitm1*0.5 + srtp1_sitp1*0.5

    srt, sit = tf.split(srt_sit, 2, axis=0)
    crt, cit = tf.split(crt_cit, 2, axis=0)

    csrt_csit = tf.switch_case(conn_type, 
                            branch_fns={
                                        0: compute_input_1freq(crt, cit, srt, sit)
                                        }
                            )

    return csrt_csit

def compute_input_1freq(cr, ci, srt, sit):

    csrt = tf.matmul(cr, srt)
    csit = tf.matmul(ci, srt)

    csrt_csit = tf.concat([csrt, csit], axis=0)

    return csrt_csit 

def crdot_cidot(crt_cit, lrn_type, lambda_, mu1, mu2, kappa, xst_yst, xtt_ytt, freqss, freqst):

    crt, cit = tf.split(crt_cit, 2, axis=0)

    cr2tplusci2t = tf.add(tf.pow(crt, 2),
                        tf.pow(cit, 2))
    cr2tplusci2t_cr2tplusci2t = tf.concat([cr2tplusci2t, 
                                cr2tplusci2t], axis=0)
    cr2tplusci2tsquared = tf.pow(cr2tplusci2t, 2)
    cr2tplusci2tsquared_cr2tplusci2tsquared = tf.concat([cr2tplusci2tsquared, 
                                cr2tplusci2tsquared], axis=0)

    dcrt_dcit = tf.switch_case(lrn_type, 
                            branch_fns={
                                        0: update_1freq(crt_cit, lambda_, mu1, mu2, kappa,epsilonc, ones, xst_yst, xtt_ytt, freqss, freqst, cr2tplusci2t_cr2tplusci2t, cr2tplusci2tsquared_cr2tplusci2tsquared)
                                        }
                            )

    return dcrdt_dcidt

def update_connection_1freq(crt_cit, lambda_, mu1, mu2, kappa, epsilonc, ones, xst_yst, xtt_ytt, freqss, freqst, cr2tplusci2t_cr2tplusci2t, cr2tplusci2tsquared_cr2tplusci2tsquared):

    xst, yst = tf.split(xst_yst, 2, axis=0)
    xtt, ytt = tf.split(xtt_ytt, 2, axis=0)
    
    xss_xss = tf.concat([xss, xss], axis=0)
    plusytt_minusxtt = tf.concat([ytt, tf.scalar_mul(-1, xtt)], axis=0) 
    yss_yss = tf.concat([yss, yss], axis=0)

    dcrdt_dcidt = tf.add_n(
                        tf.scalar_mul(lambda_, crt_cit),
                        tf.scalar_mul(mu1, tf.multiply(crt_cit,
                                                cr2tplusci2t_cr2tplusci2t)),
                        tf.divide(    
                            tf.scalar_mul(tf.multiply(epsilonc, mu2), 
                                        tf.multiply(crt_cit,
                                                cr2tplusci2tsquared_cr2tplusci2tsquared)),
                            tf.subtract(ones, 
                                tf.scalar_mul(epsilonc, tf.multiply(crt_cit,
                                                cr2tplusci2t_cr2tplusci2t)))),
                        tf.scalar_mul(kappa, tf.add(tf.multiply(xtt_ytt, 
                                                                xss_xss),
                                                    tf.multiply(plusytt_minusxtt,
                                                                yss_yss)
                                                    )
                                        )
                        )

    dcrdt_dcidt = tf.multiply(tf.divide(tf.add(freqss, freqst), 2), dcrdt_dcidt)

    return dcdt_dcit
