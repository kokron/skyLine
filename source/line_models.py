'''
Catalog of different lines and models
'''

import numpy as np
from numpy.random import normal,multivariate_normal
import astropy.units as u
import astropy.constants as cu

##############
## CO LINES ##
##############

def CO_Li16(self,SFR,pars,nu0,rng):
    '''
    Model for CO(1-0) line from Li+2016 (arXiv:1503.08833)

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

    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    LCO_samples = np.zeros(len(SFR))*u.Lsun

    #Convert halo SFR to IR luminosity
    LIR = 1e10 * SFR[inds]/delta_mf
    #Transform IR luminosity to CO luminosity (log10)
    log10_LCO = (np.log10(LIR) - beta)/alpha
    #Add normal scatter in the log10(LCO) and transform to Lsun and give units
    sigma_base_e = sigma_L*2.302585
    LCO_samples[inds] = 10**(log10_LCO)*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, log10_LCO.shape)*4.9e-5*u.Lsun*(nu0/(115.27*u.GHz))**3

    return LCO_samples


def CO_lines_scaling_LFIR(self,SFR,pars,nu0,rng):
    '''
    Returns the luminosity for CO lines lines that have empirical scaling relations with FIR luminosity

    Examples include: All the CO rotational ladder lines
    (From Kamenetzky+2016, arXiv:1508.05102)

    Relation is: log10(LFIR) = alpha*log10(LCO')+beta

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -alpha
            -beta
            -alpha_std  Std of the alpha param
            -beta_std   Std of the beta param
            -sigma_L    Scatter in dex of the luminosity
    '''
    try:
        alpha,beta,alpha_std,beta_std,sigma_L = pars['alpha'],pars['beta'],pars['alpha_std'],pars['beta_std'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for CO_lines_scaling_LFIR are "alpha","beta", "alpha_std", "beta_std","sigma_L" but {} were provided'.format(pars.keys()))

    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    L = np.zeros(len(SFR))*u.Lsun
    
    #IRX - Mstar relation from Bouwens 2020, arXiv:2009.10727
    log10Ms_IRX,alpha_IRX = multivariate_normal(np.array([9.15,0.97]),np.diag(np.array([0.17**2,0.17**2])),SFR[inds].shape)
    IRX = (self.halo_catalog['SM_HALO'][inds]/10**log10Ms_IRX)**alpha_IRX#10**-9.17*self.halo_catalog['SM_HALO'][inds]
    #We get the LIR using SFRs and IRX, with coeffiencients from Kennicutt & Evans 2012 assuming Kroupa IMG
    K_IR,K_UV = 1.49e-10, 1.71e-10 
    LIR = SFR[inds]/(K_IR + K_UV/IRX)*u.Lsun
    
    std = multivariate_normal(np.array([alpha,beta]),np.diag(np.array([alpha_std**2,beta_std**2])),LIR.shape)
    alpha_par,beta_par = std[:,0],std[:,1]

    Lp = 10**((np.log10(LIR.value)-beta_par)/alpha_par)

    Lmean = (4.9e-5*u.Lsun)*Lp*(nu0/(115.27*u.GHz))**3

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L[inds] = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)

    return L


##############
## CII LINE ##
##############

def CII_Silva15(self,SFR,pars,nu0,rng):
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

    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    L = np.zeros(len(SFR))*u.Lsun

    # LCII relation
    Lmean = 10**(aLCII*np.log10(SFR[inds])+bLCII)*u.Lsun

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L[inds] = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)

    return L

def CII_Lagache18(self,SFR,pars,nu0,rng):
    '''
    Model for CII line from Lagache+2018 (arXiv:1711.00798)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -alpha1, alpha2, beta1, beta2    Fit to log10(L_CII/Lsun) = alpha*log10(SFR/(Msun/yr)) + beta, where alpha=alpha1 + alpha2*z and beta=beta1 + beta2*z
            -sigma_L: Scatter in dex of the CII luminosity
    '''
    try:
        alpha1, alpha2, beta1, beta2, sigma_L = pars['alpha1'],pars['alpha2'],pars['beta1'],pars['beta2'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for CII_Lagache18 are alpha1, alpha2, beta1, beta2, sigma_L, but {} were provided'.format(pars.keys()))
    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    L = np.zeros(len(SFR))*u.Lsun

    # LCII relation
    alpha=alpha1+alpha2*self.halo_catalog['Z'][inds]
    beta=beta1+beta2*self.halo_catalog['Z'][inds]

    Lmean = 10**(alpha*np.log10(SFR[inds])+beta)*u.Lsun

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L[inds] = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)

    return L

##############
## Ly-alpha LINE ##
##############

def Lyalpha_Chung19(self,SFR,pars,nu0,rng):
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

##############
## 21-cm LINE ##
##############

def HI_VN18(self,SFR,pars,nu0,rng):
    '''
    Model for 21-cm line used in Villaescusa-Navarro+2018 (arXiv:1804.09180)

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -M0, Mmin, alpha    Normalization, cutoff mass, and slope in the M_HI-M_halo relation
            -sigma_L    log-normal scatter in the luminosity

    M
    '''
    try:
        M0, Mmin, alpha,sigma_L = pars['M0'],pars['Mmin'],pars['alpha'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for HI_VN18 are M0, Mmin, alpha, and sigma_L, but {} were provided'.format(pars.keys()))

    Mhalo_Msun = (self.halo_catalog['M_HALO']*self.Msunh).to(u.Msun)
    MHI=M0*np.exp(-(Mmin/Mhalo_Msun)**0.35)*(Mhalo_Msun/Mmin)**alpha

    A10=2.869e-15*u.s**(-1) #spontaneous emission coefficient
    coeff=((3/4)*A10*cu.h*self.line_nu0['HI']/cu.m_p).to(u.Lsun/u.Msun)
    LHI=coeff*MHI

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    LHI_samples = LHI*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, LHI.shape)

    return (LHI_samples).to(u.Lsun)



#####################################
## SFR Kennicutt scaling relations ##
#####################################

def SFR_scaling_relation_Kennicutt(self,SFR,pars,nu0,rng):
    '''
    Model for SFR-related lines used in Gong+2017 (arXiv:1610.09060),
    employing Kennicutt relations and extinctions.

    Examples include: Halpha, Hbeta, OII, OIII_0p5

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K            linear factor SFR = K*L (L in ergios/s)
            -Kstd         Std of the linear relation
            -Aext         Extinction of the line
            -sigma_L: Scatter in dex of the luminosity
    '''
    try:
        K,Kstd,Aext,sigma_L = pars['K'],pars['Kstd'],pars['Aext'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for SFR_scaling_relation_Kennicutt are "K","Kstd", Aext, sigma_L but {} were provided'.format(pars.keys()))
    #Spread in the linear relation
    factor = normal(K,Kstd,len(SFR))
    L = (SFR*factor*u.erg/u.s).to(u.Lsun)

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L = L*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, L.shape)

    return L*10**(-Aext/2.5)


###########################
## LIR scaling relations ##
###########################

def FIR_scaling_relation(self,SFR,pars,nu0,rng):
    '''
    Returns the luminosity for lines that have empirical scaling relations with FIR luminosity

    Examples include: OIII_51, NIII, NII, OI_63, OIII_88, OI_145, CII
    (From Spignolio+2012, arXiv:1110.4837, check the erratum for actual numbers)

    Relation is: log10(L/(1e41 erg/s)) = alpha*log10(LIR/(1e41 erg/s))-beta

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -alpha
            -beta
            -alpha_std  Std of the alpha param
            -beta_std   Std of the beta param
            -sigma_L    Scatter in dex of the luminosity
    '''
    try:
        alpha,beta,alpha_std,beta_std,sigma_L = pars['alpha'],pars['beta'],pars['alpha_std'],pars['beta_std'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for FIR_scaling_relation are "alpha","beta", "alpha_std", "beta_std","sigma_L" but {} were provided'.format(pars.keys()))
    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    L = np.zeros(len(SFR))*u.Lsun
    
    #IRX - Mstar relation from Bouwens 2017, arXiv:1606.05280
    IRX = 10**-9.17*self.halo_catalog['SM_HALO'][inds]
    #We get the LIR using SFRs and IRX, with coeffiencients from Kennicutt & Evans 2012
    K_IR,K_UV = 1.49e-10, 1.71e-10
    LIR = (SFR[inds]/(K_IR + K_UV*10**-IRX)*u.Lsun).to(u.erg/u.s)

    LIR_norm = LIR*(1/1e41)

    std = multivariate_normal(np.array([alpha,beta]),np.diag(np.array([alpha_std**2,beta_std**2])),LIR.shape)
    alpha_par,beta_par = std[:,0],std[:,1]

    Lerg_norm = 10**(alpha_par*np.log10(LIR_norm.value)-beta_par)
    Lmean = (Lerg_norm*1e41*u.erg/u.s).to(u.Lsun)

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L[inds] = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)

    return L
