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

def CO_Li16(self,SFR,pars,rng):
    '''
    Model for CO line from Li+2016 (arXiv:1503.08833)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -delta_mf:  IMF normalization
            -alpha:     power law coefficient relating IR and CO luminosities
            -beta:      Multiplicative normalization for the IR and CO luminosities
            -sigma_L: Scatter in dex of the CO luminosity
    '''
    try:
        alpha,beta,delta_mf,sigma_L = pars['alpha'],pars['beta'],pars['delta_mf'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for CO_Li16 are "alpha","beta","delta_mf" and "sigma_L", but {} were provided'.format(pars.keys()))

    #Convert halo SFR to IR luminosity
    LIR = 1e10 * SFR/delta_mf
    #Transform IR luminosity to CO luminosity (log10)
    log10_LCO = (np.log10(LIR) - beta)/alpha
    #Add normal scatter in the log10(LCO)
    sigma_base_e = sigma_L*2.302585
    LCO_samples = 10**(log10_LCO)*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, log10_LCO.shape)
    
    #transform to Lsun and give units
    return LCO_samples*4.9e-5*u.Lsun


##############
## CII LINE ##
##############

def CII_Silva15(self,SFR,pars,rng):
    '''
    Model for CII line from Silva+2015 (arXiv:1410.4808)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -aLCII,bLCII    Fit to log10(L_CII/Lsun) = aLCII*log10(SFR/(Msun/yr)) + bLCII
            -sigma_L: Scatter in dex of the CII luminosity
    '''
    try:
        aLCII,bLCII, sigma_L = pars['aLCII'],pars['bLCII'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for CII_Silva15 are "aLCII","bLCII", "sigma_L", but {} were provided'.format(pars.keys()))
    # LCII relation
    L = 10**(aLCII*np.log10(SFR)+bLCII)*u.Lsun
    
    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L = L*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, L.shape)

    return L

##############
## Ly-alpha LINE ##
##############

def Lyalpha_Chung19(self,SFR,pars,rng):
    '''
    Model for Lyman-alpha line used in Chung+2019 (arXiv:1809.04550)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -C      Conversion between SFR and Ly-alpha luminosity
            -xi, zeta, psi, z0, f0    Parametrize the escape fraction, reflecting the possibility of photons being absorbed by dust
            -sigma_L    log-normal scatter in the Ly-alpha luminosity

    M
    '''
    try:
        C,xi,zeta,psi,z0,f0,SFR0,sigma_L = pars['C'],pars['xi'],pars['zeta'],pars['psi'],pars['z0'],pars['f0'],pars['SFR0'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for Lyalpha_Chung19 are C, xi, zeta, psi, z0, f0, SFR0, and sigma_L, but {} were provided'.format(pars.keys()))

    fesc=(((1+np.exp(-xi*(self.halo_catalog['Z']-z0)))**(-zeta))*(f0+((1-f0)/(1+(SFR/SFR0)**(psi)))))**2
    LLya=C*SFR*fesc

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    LLya_samples = LLya*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, LLya.shape)

    return (LLya_samples*u.erg/u.s).to(u.Lsun)


#################
## Halpha LINE ##
#################

def Halpha_Gong17(self,SFR,pars,rng):
    '''
    Model for Halpha line used in Gong+2017 (arXiv:1610.09060)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K_Halpha       linear factor SFR = K_Halpha*L_Halpha (L in ergios/s)
            -Kstd_Halpha    Std of the linear relation
            -Aext_Halpha    Extinction of the Halpha line
            -sigma_L: Scatter in dex of the Halpha luminosity
    '''
    try:
        K_Halpha,Kstd_Halpha,Aext_Halpha, sigma_L = pars['K_Halpha'],pars['Kstd_Halpha'],pars['Aext_Halpha'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for Halpha_Gong17 are "K_Halpha","Kstd_Halpha", Aext_Halpha, sigma_L but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K_Halpha,Kstd_Halpha,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)
    
    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L = L*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, L.shape)

    return L*10**(-Aext_Halpha/2.5)


################
## Hbeta LINE ##
################

def Hbeta_Gong17(self,SFR,pars,rng):
    '''
    Model for Hbeta line used in Gong+2017 (arXiv:1610.09060)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K_Hbeta       linear factor SFR = K_Hbeta*L_Hbeta (L in ergios/s)
            -Kstd_Hbeta    Std of the linear relation
            -Aext_Hbeta    Extinction of the Hbeta line
            -sigma_L: Scatter in dex of the Hbeta luminosity
    '''
    try:
        K_Hbeta,Kstd_Hbeta,Aext_Hbeta, sigma_L = pars['K_Hbeta'],pars['Kstd_Hbeta'],pars['Aext_Hbeta'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for Hbeta_Gong17 are "K_Hbeta","Kstd_Hbeta", Aext_Hbeta, sigma_L but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K_Hbeta,Kstd_Hbeta,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)
    
    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L = L*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, L.shape)

    return L*10**(-Aext_Hbeta/2.5)



##############
## OII LINE ##
##############

def OII_Gong17(self,SFR,pars,rng):
    '''
    Model for OII line used in Gong+2017 (arXiv:1610.09060)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K_OII       linear factor SFR = K_OII*L_OII (L in ergios/s)
            -Kstd_OII    Std of the linear relation
            -Aext_OII    Extinction of the OII line
            -sigma_L: Scatter in dex of the OII luminosity
    '''
    try:
        K_OII,Kstd_OII,Aext_OII, sigma_L = pars['K_OII'],pars['Kstd_OII'],pars['Aext_OII'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for OII_Gong17 are "K_OII","Kstd_OII", Aext_OII, sigma_L but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K_OII,Kstd_OII,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)
    
    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L = L*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, L.shape)

    return L*10**(-Aext_OII/2.5)


###############
## OIII LINE ##
###############

def OIII_Gong17(self,SFR,pars,rng):
    '''
    Model for OIII line used in Gong+2017 (arXiv:1610.09060)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K_OIII       linear factor SFR = K_OIII*L_OIII (L in ergios/s)
            -Kstd_OIII    Std of the linear relation
            -Aext_OIII    Extinction of the OIII line
            -sigma_L: Scatter in dex of the OIII luminosity
    '''
    try:
        K_OIII,Kstd_OIII,Aext_OIII, sigma_L = pars['K_OIII'],pars['Kstd_OIII'],pars['Aext_OIII'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for OIII_Gong17 are "K_OIII","Kstd_OIII", Aext_OIII, sigma_L but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K_OIII,Kstd_OIII,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)
    
    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L = L*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, L.shape)

    return L*10**(-Aext_OIII/2.5)
