import tensorflow as tf
import numpy as np

def xdot_ydot(xt_yt, alpha, beta1, beta2, epsilon, ones, freqs, lrn_stim_connmat_params_zip):

    xt, yt = tf.split(xt_yt, 2, axis=0)
    minusyt_plusxt = tfconcat(tf.scalar_mul(-1,yt), xt, axis=0)

    x2tplusy2t = tf.add(tf.pow(xt, 2),
                        tf.pow(yt, 2))
    x2tplusy2t_x2tplusy2t = tf.concat(x2tplusy2t, 
                                x2tplusy2t, axis=0)
    x2tplusy2tsquared = tf.pow(x2tplusy2t, 2)
    x2tplusy2tsquared_x2tplusy2tsquared = tf.concat(x2tplusy2tsquared, 
                                x2tplusy2tsquared, axis=0)

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

    csrt_csit = tf.add_n([compute_inputs(lrn_stim_connmat_params) for lrn_stim_connmat_params in list(lrn_stim_connmat_params_zip)])

    dxdt_dydt = tf.multiply(freqs, tf.add(xtnew_ytnew, csrt_csit))

    return dxdt_dydt

def compute_inputs(lrn_stim_connmat_params):

    lrn_type, srt_sit, crt_cit, params = lrn_stim_connmat_params

    srt, sit = tf.split(srt_sit, 2, axis=0)
    crt, cit = tf.split(crt_cit, 2, axis=0)

    csrt_csit = tf.switch_case(lrn_type, 
                            branch_fns={
                                        0: compute_input_1freq(crt, cit, srt, sit)
                                        }
                            )

    return csrt_csit

def compute_input_1freq(cr, ci, srt, sit):

    csrt = tf.matmul(cr, srt)
    csit = tf.matmul(ci, srt)

    csrt_csit = tf.concat(csrt, csit, axis=0)

    return csrt_csit 

def crdot_cidot(crt_cit, lrn_type, lambda_, mu1, mu2, kappa, xst_yst, xtt_ytt, freqss, freqst):

    crt, cit = tf.split(crt_cit, 2, axis=0)

    cr2tplusci2t = tf.add(tf.pow(crt, 2),
                        tf.pow(cit, 2))
    cr2tplusci2t_cr2tplusci2t = tf.concat(cr2tplusci2t, 
                                cr2tplusci2t, axis=0)
    cr2tplusci2tsquared = tf.pow(cr2tplusci2t, 2)
    cr2tplusci2tsquared_cr2tplusci2tsquared = tf.concat(cr2tplusci2tsquared, 
                                cr2tplusci2tsquared, axis=0)

    dcrt_dcit = tf.switch_case(lrn_type, 
                            branch_fns={
                                        0: update_1freq(crt_cit, lambda_, mu1, mu2, kappa,epsilonc, ones, xst_yst, xtt_ytt, freqss, freqst, cr2tplusci2t_cr2tplusci2t, cr2tplusci2tsquared_cr2tplusci2tsquared)
                                        }
                            )

    return dcrdt_dcidt

def update_connection_1freq(crt_cit, lambda_, mu1, mu2, kappa, epsilonc, ones, xst_yst, xtt_ytt, freqss, freqst, cr2tplusci2t_cr2tplusci2t, cr2tplusci2tsquared_cr2tplusci2tsquared):

    xst, yst = tf.split(xst_yst, 2, axis=0)
    xtt, ytt = tf.split(xtt_ytt, 2, axis=0)
    
    xss_xss = tf.concat(xss, xss, axis=0)
    plusytt_minusxtt = tf.concat(ytt, tf.scalar_mul(-1, xtt), axis=0) 
    yss_yss = tf.concat(yss, yss, axis=0)

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
