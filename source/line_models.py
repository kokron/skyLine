'''
Catalog of different lines and models
'''

import numpy as np
import astropy.units as u
import astropy.constants as cu

def CO_Li16(self,SFR,pars):
    '''
    Model for CO line from Li+2016 (arXiv:1503.08833)
    
    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -delta_mf:  IMF normalization
        -alpha:     power law coefficient relating IR and CO luminosities
        -beta:      Multiplicative normalization for the IR and CO luminosities
        -sigma_LCO: Scatter in dex of the CO luminosity
    '''
    try:
        alpha,beta,delta_mf,sigma_LCO = pars['alpha'],pars['beta'],pars['delta_mf'],pars['sigma_LCO']
    except:
        raise ValueError('The model_pars for CO_Li16 are "alpha","beta","delta_mf" and "sigma_LCO", but {} were provided'.format(pars.keys()))
    
    #Convert halo SFR to IR luminosity
    LIR = 1e10 * SFR/delta_mf
    #Transform IR luminosity to CO luminosity (log10)
    log10_LCO = (np.log10(LIR) - beta)/alpha
    #Add normal scatter in the log10(LCO)
    LCO_samples = 10**(np.random.normal(log10_LCO, sigma_LCO))
    #transform to Lsun and give units
    LCO_samples *= 4.9e-5*u.Lsun
    
    return LCO_samples
    
