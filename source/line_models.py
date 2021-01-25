'''
Catalog of different lines and models
'''

import numpy as np
from numpy.random import normal
import astropy.units as u
import astropy.constants as cu

#############
## CO LINE ##
#############

def CO_Li16(self,SFR,pars):
    '''
    Model for CO line from Li+2016 (arXiv:1503.08833)
    
    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
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
    return LCO_samples*4.9e-5*u.Lsun
    
    
##############
## CII LINE ##
##############

def CII_Silva15(self,SFR,pars):
    '''
    Model for CII line from Silva+2015 (arXiv:1410.4808)
    
    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -aLCII,bLCII    Fit to log10(L_CII/Lsun) = aLCII*log10(SFR/(Msun/yr)) + bLCII
    '''
    try:
        a,b = pars['aLCII'],pars['bLCII']
    except:
        raise ValueError('The model_pars for CII_Silva15 are "aLCII","bLCII", but {} were provided'.format(pars.keys()))
    # LCII relation
    L = 10**(aLCII*np.log10(SFR/(1*u.Msun/u.yr))+bLCII)*u.Lsun
    
    return L
    
    
#################
## Halpha LINE ##
#################

def Halpha_Gong17(self,SFR,pars):
    '''
    Model for Halpha line used in Gong+2017 (arXiv:1610.09060)
    
    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K_Halpha       linear factor SFR = K_Halpha*L_Halpha (L in ergios/s)
            -Kstd_Halpha    Std of the linear relation
            -Aext_Halpha    Extinction of the Halpha line
    '''
    try:
        K_Halpha,Kstd_Halpha,Aext_Halpha = pars['K_Halpha'],pars['Kstd_Halpha'],pars['Aext_Halpha']
    except:
        raise ValueError('The model_pars for Halpha_Gong17 are "K_Halpha","Kstd_Halpha", Aext_Halpha, but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K_Halpha,Kstd_Halpha,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)

    return L*10**(-Aext_Halpha/2.5)
    

################
## Hbeta LINE ##
################

def Hbeta_Gong17(self,SFR,pars):
    '''
    Model for Hbeta line used in Gong+2017 (arXiv:1610.09060)
    
    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K_Hbeta       linear factor SFR = K_Hbeta*L_Hbeta (L in ergios/s)
            -Kstd_Hbeta    Std of the linear relation
            -Aext_Hbeta    Extinction of the Hbeta line
    '''
    try:
        K_Hbeta,Kstd_Hbeta,Aext_Hbeta = pars['K_Hbeta'],pars['Kstd_Hbeta'],pars['Aext_Hbeta']
    except:
        raise ValueError('The model_pars for Hbeta_Gong17 are "K_Hbeta","Kstd_Hbeta", Aext_Hbeta, but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K_Hbeta,Kstd_Hbeta,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)

    return L*10**(-Aext_Hbeta/2.5)



##############
## OII LINE ##
##############

def OII_Gong17(self,SFR,pars):
    '''
    Model for OII line used in Gong+2017 (arXiv:1610.09060)
    
    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K_OII       linear factor SFR = K_OII*L_OII (L in ergios/s)
            -Kstd_OII    Std of the linear relation
            -Aext_OII    Extinction of the OII line
    '''
    try:
        K_OII,Kstd_OII,Aext_OII = pars['K_OII'],pars['Kstd_OII'],pars['Aext_OII']
    except:
        raise ValueError('The model_pars for OII_Gong17 are "K_OII","Kstd_OII", Aext_OII, but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K_OII,Kstd_OII,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)

    return L*10**(-Aext_OII/2.5)
    

###############
## OIII LINE ##
###############

def OIII_Gong17(self,SFR,pars):
    '''
    Model for OIII line used in Gong+2017 (arXiv:1610.09060)
    
    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K_OIII       linear factor SFR = K_OIII*L_OIII (L in ergios/s)
            -Kstd_OIII    Std of the linear relation
            -Aext_OIII    Extinction of the OIII line
    '''
    try:
        K_OIII,Kstd_OIII,Aext_OIII = pars['K_OIII'],pars['Kstd_OIII'],pars['Aext_OIII']
    except:
        raise ValueError('The model_pars for OIII_Gong17 are "K_OIII","Kstd_OIII", Aext_OIII, but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K_OIII,Kstd_OIII,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)

    return L*10**(-Aext_OIII/2.5)
    
    
    
    
    
    
    
    
    
    
    
    
