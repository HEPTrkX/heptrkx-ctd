import theano as th
import theano.tensor as T
import theano.tensor.nlinalg as L

def covariance_from_network_params(params):
    """Parameterization of a 2x2 covariance matrix using three 
        real numbers: log(variance1), log(variance2), 
        and arctanh(correlation).
        Input: vector of length 3
        Output: vector [variance1, covariance, variance2]"""
    variance1 = T.exp(params[0])
    variance2 = T.exp(params[1])
    corr = T.tanh(params[2])
    return T.stack( [ variance1, corr*T.sqrt(variance1*variance2), variance2 ] )

def _covariance_from_network_outputs(outputs):
    """Runs covariance_from_network_params on each event in the batch"""
    return th.scan( fn=covariance_from_network_params,
            sequences=[outputs])[0]

def covariance_from_network_outputs(outputs):
    """Parameterization of a 2x2 covariance matrix using three 
        real numbers.
        Input: tensor of shape (batch_size, num_tracks, 3)
        Output: tensor of the same shape, where the three 
            parameters represent [variance1, covariance, variance2]"""
    return th.scan( fn=_covariance_from_network_outputs,
            sequences=[outputs])[0]

def covariance_matrix_2D(params):
    """creates a 2D covariance matrix from a 1D array of 3 parameters:
        [variance1, covariance, variance2]"""
    x = T.stack( [params, T.roll(params, shift=-1)] )
    return x[:,:2]

def minus_two_log_gauss_likelihood_2D(residuals, covariance_values):
    """computes the -2 log gaussian likelihood (ignoring the constant term)
        from the given residuals and covariance values in the form of a 
        1D array: [variance1, covariance, variance2]"""
    cov = covariance_matrix_2D(covariance_values)
    det = L.Det()(cov)
    precis = L.MatrixInverse()(cov)
    term1 = T.dot( T.transpose(residuals), T.dot( precis, residuals ))
    term2 = T.log( det )
    return term1 + term2

def gauss_likelihood_loss_2D(y_true, y_pred):
    """
    Computes negative log gaussian likelihood over the track
    parameters and their uncertainties.  
    Format for y_pred:
        axis 0: events in batch
        axis 1: tracks in event
        axis 2: [track slope, track intercept]
            +[ slope variance, covariance, intercept variance ]
    y_true has the same format, except that the track covariance
    matrix parameters are not defined for the ground truth tracks.
    Only the slopes and intercepts are used; the other entries are ignored.
    """
    true_params = y_true[:,:,:2]
    pred_params = y_pred[:,:,:2]
    resids = true_params-pred_params
    cov_values = y_pred[:,:,2:]

    unrolled_resids = T.reshape(resids, (-1, 2))
    unrolled_cov_values = T.reshape(cov_values, (-1, 3))
    lls, _ = th.scan( fn=minus_two_log_gauss_likelihood_2D, 
            sequences=[unrolled_resids, unrolled_cov_values] )
    return T.sum( lls ) 
