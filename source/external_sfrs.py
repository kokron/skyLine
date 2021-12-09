'''
Catalog for external SFRs
'''

import numpy as np
from scipy.interpolate import griddata 
import os

def Behroozi_SFR(M, z):
    '''
    Returns SFR(M,z) interpolated from Behroozi et al. 2013
    '''

    return SFR_Mz_2dinterp(M,z,'sfr_table_Behroozi.dat')

def UniverseMachine_SFR(M,z):
    '''
    Returns SFR(M,z) interpolated from tables obtained from Universe Machine best-fit
    realization
    '''

    return SFR_Mz_2dinterp(M,z,'UM_sfr.dat')

def Custom_SFR(M,z,pars):
    '''
    Returns SFR using a double power law with and additional power to control width of peak
    '''
    try:
        M0,Ma,Mb,a,b,c = pars['M0'],pars['Ma'],pars['Mb'],pars['a'],pars['b'],pars['c']
    except:
        raise ValueError('The model_pars for Custom_SFR are M0,Ma,Mb,a,b,c, but {} were provided'.format(pars.keys()))
    return M0*((M/Ma)**a)*(1+(M/Mb)**b)**c

def Dongwoo_SFR(M,z,pars):
    '''
    Returns SFR using a doble power law following Chung et al 2021 parameterization.
    '''

    try:
        A, B, C, M_h = pars['A'],pars['B'],pars['C'],pars['M_h']
    except:
        raise ValueError('The model_pars for Dongwoo_SFR are A, B, C, M_h but {} were provided'.format(pars.keys()))
    ratio = M/M_h

    return C/(ratio**A + ratio**B)
################################

def SFR_Mz_2dinterp(M,z,SFR_file):
    '''
    Returns SFR(M,z) interpolated from tables of 1+z, log10(Mhalo/Msun) and 
    log10(SFR / (Msun/yr)), in three columns, where 1+z is the innermost index 
    (the one running fast compared with the mass)
    '''
    SFR_folder = os.path.dirname(os.path.realpath(__file__)).split("source")[0]+'SFR_tables/'
    try:
        x = np.loadtxt(SFR_folder+SFR_file)
    except:
        x = np.loadtxt(SFR_file)
    zb = np.unique(x[:,0])-1.
    logMb = np.unique(x[:,1])
    logSFRb = x[:,2].reshape(len(zb),len(logMb),order='F')
   


    xx, yy = np.meshgrid(logMb, zb)

    ingrid = np.array([xx, yy]).reshape(2, len(zb)*len(logMb))

    inval = logSFRb.reshape(len(zb)*len(logMb))
    #Assuming in Msun/h units
    logM = np.log10((M))
    
    grid = np.array([logM, z]) 
    logSFR_interp = griddata(ingrid.T, inval, grid.T, fill_value=-40.)
    SFR = 10**logSFR_interp
    return SFR
