'''
Catalog for external SFRs
'''

import numpy as np
from scipy.interpolate import interp2d
import os

def Behroozi_SFR(M, z):
    '''
    Returns SFR(M,z) interpolated from Behroozi et al.
    '''
    SFR_folder = os.path.dirname(os.path.realpath(__file__)).split("source")[0]+'SFR_tables/'
    x = np.loadtxt(SFR_folder+'sfr_table_Behroozi.dat')
    zb = np.unique(x[:,0])-1.
    logMb = np.unique(x[:,1])
    logSFRb = x[:,2].reshape(137,122,order='F')
    
    logSFR_interp = interp2d(logMb,zb,logSFRb,bounds_error=False,fill_value=0.)
    
    logM = np.log10((M))
    SFR = np.zeros(logM.size)
    for ii in range(0,logM.size):
        SFR[ii] = 10.**logSFR_interp(logM[ii],z[ii])
    
    return SFR
