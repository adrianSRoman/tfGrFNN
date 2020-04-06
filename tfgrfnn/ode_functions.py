import tensorflow as tf
import numpy as np

def xdot_ydot(t, xt_yt, connmats_state, connections, sources_state, alpha=None, beta1=None, beta2=None, epsilon=None, freqs= None):

    omega = tf.constant(2*np.pi, dtype=tf.float32)

    xt, yt = tf.split(xt_yt, 2, axis=1)

    x2tplusy2t = tf.add(tf.pow(xt, 2),
                        tf.pow(yt, 2))
    x2tplusy2tsquared = tf.pow(x2tplusy2t, 2)

    xtnew = tf.add_n([tf.multiply(alpha, xt),
                        tf.multiply(omega, tf.multiply(-1.0, yt)),
                        tf.multiply(beta1, tf.multiply(xt, x2tplusy2t)),
                        tf.divide(
                            tf.multiply(tf.multiply(epsilon,beta2),
                                tf.multiply(xt, x2tplusy2tsquared)),
                            tf.subtract(tf.constant(1.0, dtype=tf.float32),
                                tf.multiply(epsilon, x2tplusy2t)))])

    ytnew = tf.add_n([tf.multiply(alpha, yt),
                        tf.multiply(omega, xt),
                        tf.multiply(beta1, tf.multiply(yt, x2tplusy2t)),
                        tf.divide(
                            tf.multiply(tf.multiply(epsilon,beta2),
                                tf.multiply(yt, x2tplusy2tsquared)),
                            tf.subtract(tf.constant(1.0, dtype=tf.float32),
                                tf.multiply(epsilon, x2tplusy2t)))])

    xtnew_ytnew = tf.concat([xtnew, ytnew], axis=1)

    csrt_csit = tf.add_n([compute_input(connmat_state, 
                            sources_state[connections[iconn].sourceintid],
                            xt_yt, connections[iconn].params['typeint'], epsilon,
                            connections[iconn].params['weight'])
                        for iconn, connmat_state in enumerate(connmats_state)]) if connmats_state else 0
            
    dxdt_dydt = tf.multiply(freqs, tf.add(xtnew_ytnew, csrt_csit))
    
    return dxdt_dydt


def compute_input(connmat_state, source_state, target_state, typeint, epsilon, input_weight):

    def compute_input_1freq(srt_sit=source_state, crt_cit=connmat_state, input_weight=input_weight):

        srt, sit = tf.split(srt_sit, 2, axis=1)
        crt, cit = tf.split(crt_cit, 2, axis=1)
        csrt = tf.matmul(srt, crt) - tf.matmul(sit, cit)
        csit = tf.matmul(sit, crt) + tf.matmul(srt, cit)
        csrt = tf.multiply(input_weight, csrt)
        csit = tf.multiply(input_weight, csit)

        csrt_csit = tf.concat([csrt, csit], axis=1)

        return csrt_csit 

    def compute_input_allfreq(srt_sit=source_state, trt_tit=target_state, 
                                crt_cit=connmat_state, epsilon=epsilon, input_weight=input_weight):

        srt, sit = tf.split(srt_sit, 2, axis=1)
        trt, tit = tf.split(trt_tit, 2, axis=1)
        crt, cit = tf.split(crt_cit, 2, axis=1)

        sqrteps = tf.sqrt(tf.subtract(epsilon,0.05))
        sr2 = tf.pow(srt, 2)
        si2 = tf.pow(sit, 2)
        ti2 = tf.pow(tit, 2)

        one_min_re2 = tf.pow(tf.subtract(tf.constant(1.0, dtype=tf.float32),
                                    tf.multiply(srt, sqrteps)), 2)
        Pdenominator = tf.add(one_min_re2, tf.multiply(si2, epsilon))
        Pn1r = tf.divide(tf.add_n([tf.multiply(-sr2, sqrteps),
                                    srt,
                                    tf.multiply(-si2, sqrteps)]),
                        Pdenominator)
        Pn1i = tf.divide(sit,
                        Pdenominator)
        Pn2r = tf.divide(tf.add(tf.constant(1.0, dtype=tf.float32),
                                tf.multiply(-srt,sqrteps)),
                        Pdenominator)
        Pn2i = tf.divide(-tf.multiply(sit, sqrteps),
                        Pdenominator)

        one_min_re2 = tf.pow(tf.subtract(tf.constant(1.0, dtype=tf.float32),
                                    tf.multiply(trt, sqrteps)), 2)
        Adenominator = tf.add(one_min_re2, tf.multiply(ti2, epsilon))
        Ar = tf.divide(tf.add(tf.constant(1.0, dtype=tf.float32),
                                tf.multiply(-trt,sqrteps)),
                        Adenominator)
        Ai = tf.divide(-tf.multiply(tit, sqrteps),
                        Adenominator)

        Pnr = tf.multiply(Pn1r,Pn2r) - tf.multiply(Pn1i,Pn2i) 
        Pni = tf.multiply(Pn1i,Pn2r) + tf.multiply(Pn1r,Pn2i) 

        csrt = tf.matmul(Pnr,crt) - tf.matmul(Pni,cit)
        csit = tf.matmul(Pni,crt) + tf.matmul(Pnr,cit)

        csrt = tf.multiply(Ar,csrt) - tf.multiply(Ai,csit)
        csit = tf.multiply(Ai,csrt) + tf.multiply(Ar,csit)

        csrt = tf.multiply(input_weight, csrt)
        csit = tf.multiply(input_weight, csit)

        csrt_csit = tf.concat([csrt, csit], axis=1)

        return csrt_csit 

    csrt_csit = tf.switch_case(typeint,
                                branch_fns={0: compute_input_1freq,
                                            1: compute_input_allfreq})

    return csrt_csit
    
def crdot_cidot(t, xst_yst, crt_cit, xtt_ytt, params):

    def nolearning(crt_cit=crt_cit):
        
        return tf.constant(0, dtype=tf.float32, shape=crt_cit.shape)

    '''
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
    '''

    if params['learn']:
        dcrdt_dcidt = nolearning(crt_cit) 
    #    learntype = params['typeint']
    #    dcrdt_dcidt = tf.switch_case(learntype,
    #                                    branch_fns={0: learn_1freq,
    #                                                1: learn_allfreq})
    else:
        dcrdt_dcidt = nolearning(crt_cit) 

    return dcrdt_dcidt
