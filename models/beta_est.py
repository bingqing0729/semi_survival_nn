import numpy as np
import scipy.optimize as spo
# Z is p-dimensional
def Beta_est(ind,fail,z,dt,cum_g_pred,ind_v,fail_v,z_v,dt_v,cum_g_pred_v):
    p = 2
    # define object function
    def BF(*args):
        b = args[0]
        Loss_F = np.sum(ind*(-fail * (z*b[1]+b[0] + cum_g_pred) + np.exp(z*b[1]+b[0] + cum_g_pred)*dt))/z.shape[0]
        Loss_F_V = np.sum(ind_v*(-fail_v * (z_v*b[1]+b[0] + cum_g_pred_v) + np.exp(z_v*b[1]+b[0] + cum_g_pred_v)*dt_v))/z_v.shape[0]
        return Loss_F + Loss_F_V
    result = spo.minimize(BF,np.zeros(p),method='SLSQP')
    return result['x']
