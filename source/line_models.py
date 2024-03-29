'''
Catalog of different lines and models
'''

import numpy as np
from numpy.random import normal,multivariate_normal
import astropy.units as u
import astropy.constants as cu

######################
## Lines considered ##
######################

def lines_included(self):
    '''
    Returns a dictionary with the lines that are considered within the code.
    Add here a line and the rest-frame frequency if required
    '''
    lines = dict(CO_J10 = 115.271*u.GHz, CO_J21 = 2*115.271*u.GHz, CO_J32 = 3*115.271*u.GHz,CO_J43 = 4*115.271*u.GHz,
                        CO_J54 = 5*115.271*u.GHz, CO_J65 = 6*115.271*u.GHz, CO_J76 = 7*115.271*u.GHz,
                        CII = 1900.539*u.GHz, HI = 1.4204134*u.GHz,NIII = 5230.1545*u.GHz,NII = 2459.3311*u.GHz,
                        OIII_88 = 3392.8526*u.GHz, OI_63 = 4745.0531*u.GHz, OI_145 = 2060.4293*u.GHz, 
                        Lyalpha = 2465398.5*u.GHz, Halpha = 456805.72*u.GHz, Hbeta = 616730.01028595*u.GHz,
                        OII = 804380.08585994*u.GHz, OIII_0p5 = 598746.67066107*u.GHz)
    return lines

###################
## IR Luminosity ##
###################

def LIR(self,halos,SFR,pars,rng):
    '''
    Obtain the IR luminosity from SFR or stellar mass

    Parameters:
        -halos:     Halos to take into account (with all halo props)
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters
            -IRX_name:      The reference to use the IRX from
            -IRX_params:    The parameters required to compute the IRX
                            (check each if instance below)
            -K_IR, K_UV:    The coefficients to relate SFR to L_IR and L_UV
    '''
    #avoid SFR=0 issues
    inds = np.where(SFR>0)[0]

    #Try to get IRX:
    if 'IRX_name' not in pars:
        IRX = 1 #dummy value
        K_IR = pars['K_IR']
        K_UV = 0.
    else:
        #IRX - Mstar relation from Bouwens 2016, arXiv:1606.05280
        if pars['IRX_name'] == 'Bouwens2016':
            log10IRX_0,sigma_IRX = pars['log10IRX_0'],pars['sigma_IRX']
            IRX = 10**log10IRX_0*halos['SM_HALO'][inds]
        #IRX - Mstar relation from Bouwens 2020, arXiv:2009.10727
        elif pars['IRX_name'] == 'Bouwens2020':
            log10Ms_IRX,alpha_IRX,sigma_IRX = pars['log10Ms_IRX'],pars['alpha_IRX'],pars['sigma_IRX']
            IRX = (halos['SM_HALO'][inds]/10**log10Ms_IRX)**alpha_IRX
        #IRX - Mstar relation from Heinis 2014, arXiv:1310.3227
        elif pars['IRX_name'] == 'Heinis2014':
            log10Ms_IRX,alpha_IRX,log10IRX_0,sigma_IRX = pars['log10Ms_IRX'],pars['alpha_IRX'],pars['log10IRX_0'],pars['sigma_IRX']
            IRX = (halos['SM_HALO'][inds]/10**log10Ms_IRX)**alpha_IRX*10**log10IRX_0
        else:
            raise ValueError('Please choose a valid IRX model')
        #Add mean-preserving lognormal scatter in the IRX relation
        sigma_base_e = sigma_IRX*2.302585
        IRX = IRX*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, IRX.shape)

        K_IR,K_UV = pars['K_IR'],pars['K_UV']

    LIR = SFR[inds]/(K_IR + K_UV/IRX)*u.Lsun

    return LIR
    

def LIR_and_LUV(self,halos,SFR,pars,rng):
    '''
    Obtain the IR and UV luminosities from SFR or stellar mass
    
    -to use outside code

    Parameters:
        -halos:     Halos to take into account (with all halo props)
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters
            -IRX_name:      The reference to use the IRX from
            -IRX_params:    The parameters required to compute the IRX
                            (check each if instance below)
            -K_IR, K_UV:    The coefficients to relate SFR to L_IR and L_UV
    '''
    #avoid SFR=0 issues
    inds = np.where(SFR>0)[0]

    #Try to get IRX:
    if 'IRX_name' not in pars:
        raise ValueError('To use this function you need to choose an IRX model')
    else:
        #IRX - Mstar relation from Bouwens 2016, arXiv:1606.05280
        if pars['IRX_name'] == 'Bouwens2016':
            log10IRX_0,sigma_IRX = pars['log10IRX_0'],pars['sigma_IRX']
            IRX = 10**log10IRX_0*halos['SM_HALO'][inds]
        #IRX - Mstar relation from Bouwens 2020, arXiv:2009.10727
        elif pars['IRX_name'] == 'Bouwens2020':
            log10Ms_IRX,alpha_IRX,sigma_IRX = pars['log10Ms_IRX'],pars['alpha_IRX'],pars['sigma_IRX']
            IRX = (halos['SM_HALO'][inds]/10**log10Ms_IRX)**alpha_IRX
        #IRX - Mstar relation from Heinis 2014, arXiv:1310.3227
        elif pars['IRX_name'] == 'Heinis2014':
            log10Ms_IRX,alpha_IRX,log10IRX_0,sigma_IRX = pars['log10Ms_IRX'],pars['alpha_IRX'],pars['log10IRX_0'],pars['sigma_IRX']
            IRX = (halos['SM_HALO'][inds]/10**log10Ms_IRX)**alpha_IRX*10**log10IRX_0
        else:
            raise ValueError('Please choose a valid IRX model')
        #Add mean-preserving lognormal scatter in the IRX relation
        sigma_base_e = sigma_IRX*2.302585
        IRX = IRX*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, IRX.shape)

        K_IR,K_UV = pars['K_IR'],pars['K_UV']

    LIR = SFR[inds]/(K_IR + K_UV/IRX)*u.Lsun

    return LIR,LIR/IRX

##############
## CO LINES ##
##############

def CO_Li16(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Model for CO(1-0) line from Li+2016 (arXiv:1503.08833)

    Parameters:
        -halos:     Halos to take into account (with all halo props)
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

    #Convert halo SFR to IR luminosity following Li+2016 instead of our own L_IR
    L_IR = 1e10 * SFR[inds]/delta_mf
    #Transform IR luminosity to CO luminosity (log10)
    log10_LCO = (np.log10(L_IR) - beta)/alpha
    #Add normal scatter in the log10(LCO) and transform to Lsun and give units
    sigma_base_e = sigma_L*2.302585
    LCO_samples[inds] = 10**(log10_LCO)*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, log10_LCO.shape)*4.9e-5*u.Lsun*(nu0/(115.27*u.GHz))**3

    return LCO_samples


def CO_lines_scaling_LFIR(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Returns the luminosity for CO lines lines that have empirical scaling relations with FIR luminosity

    Examples include: All the CO rotational ladder lines
    (From Kamenetzky+2016, arXiv:1508.05102)

    Relation is: log10(LFIR) = alpha*log10(LCO')+beta

    Parameters:
        -halos:     Halos to take into account (with all halo props)
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -alpha
            -beta
            -sigma_L    Scatter in dex of the luminosity
    '''
    try:
        alpha,beta,sigma_L = pars['alpha'],pars['beta'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for CO_lines_scaling_LFIR are "alpha","beta", "sigma_L" but {} were provided'.format(pars.keys()))

    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    L = np.zeros(len(SFR))*u.Lsun

    Lp = 10**((np.log10(LIR.value)-beta)/alpha)

    Lmean = (4.9e-5*u.Lsun)*Lp*(nu0/(115.27*u.GHz))**3

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L[inds] = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)

    return L


##############
## CII LINE ##
##############

def CII_Silva15(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Model for CII line from Silva+2015 (arXiv:1410.4808)

    Parameters:
        -halos:     Halos to take into account (with all halo props)
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

def CII_Lagache18(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Model for CII line from Lagache+2018 (arXiv:1711.00798)

    Parameters:
        -halos:     Halos to take into account (with all halo props)
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
    alpha=alpha1+alpha2*halos['Z'][inds]
    beta=beta1+beta2*halos['Z'][inds]

    Lmean = 10**(alpha*np.log10(SFR[inds])+beta)*u.Lsun

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L[inds] = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)

    return L

##############
## Ly-alpha LINE ##
##############

def Lyalpha_Chung19(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Model for Lyman-alpha line used in Chung+2019 (arXiv:1809.04550)

    Parameters:
        -halos:     Halos to take into account (with all halo props)
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

    fesc=(((1+np.exp(-xi*(halos['Z']-z0)))**(-zeta))*(f0+((1-f0)/(1+(SFR/SFR0)**(psi)))))**2
    LLya=C*SFR*fesc

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    LLya_samples = LLya*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, LLya.shape)

    return (LLya_samples*u.erg/u.s).to(u.Lsun)

##############
## 21-cm LINE ##
##############

def HI_VN18(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Model for 21-cm line used in Villaescusa-Navarro+2018 (arXiv:1804.09180)

    Parameters:
        -halos:     Halos to take into account (with all halo props)
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -M0, Mmin, alpha    Normalization, cutoff mass, and slope in the M_HI-M_halo relation
            -sigma_L    log-normal scatter in the luminosity

    M
    '''
    try:
        M0, Mmin, alpha, sigma_MHI = pars['M0'],pars['Mmin'],pars['alpha'],pars['sigma_MHI']
    except:
        raise ValueError('The model_pars for HI_VN18 are M0, Mmin, alpha, and sigma_MHI, but {} were provided'.format(pars.keys()))

    Mhalo_Msun = (halos['M_HALO']*self.Msunh).to(u.Msun)
    MHI=M0*np.exp(-(Mmin/Mhalo_Msun)**0.35)*(Mhalo_Msun/Mmin)**alpha
    
    #Add scatter to the MHI relation
    sigma_base_e = sigma_MHI*2.302585
    MHI_samples = MHI*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, MHI.shape)


    A10=2.869e-15*u.s**(-1) #spontaneous emission coefficient
    coeff=((3/4)*A10*cu.h*self.line_nu0['HI']/cu.m_p).to(u.Lsun/u.Msun)
    LHI=coeff*MHI_samples
    return (LHI).to(u.Lsun)



#####################################
## SFR Kennicutt scaling relations ##
#####################################

def SFR_scaling_relation_Kennicutt(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Model for SFR-related lines used in Gong+2017 (arXiv:1610.09060),
    employing Kennicutt relations and extinctions.

    Examples include: Halpha, Hbeta, OII, OIII_0p5

    Parameters:
        -halos:     Halos to take into account (with all halo props)
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -K            linear factor SFR = K*L (L in ergios/s)
            -Aext         Extinction of the line
            -sigma_L: Scatter in dex of the luminosity
    '''
    try:
        K,Aext,sigma_L = pars['K'],pars['Aext'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for SFR_scaling_relation_Kennicutt are "K", Aext, sigma_L but {} were provided'.format(pars.keys()))
    L = (SFR*K*u.erg/u.s).to(u.Lsun)

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L = L*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, L.shape)

    return L*10**(-Aext/2.5)


###########################
## LIR scaling relations ##
###########################

def FIR_scaling_relation(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Returns the luminosity for lines that have empirical scaling relations with FIR luminosity

    Examples include: OIII_51, NIII, NII, OI_63, OIII_88, OI_145, CII
    (From Spignolio+2012, arXiv:1110.4837, check the erratum for actual numbers)

    Relation is: log10(L/(1e41 erg/s)) = alpha*log10(LIR/(1e41 erg/s))-beta

    Parameters:
        -halos:     Halos to take into account (with all halo props)
        -SFR:       SFR of the halo in Msun/yr
        -pars:      Dictionary of parameters for the model
            -alpha
            -beta
            -sigma_L    Scatter in dex of the luminosity
    '''
    try:
        alpha,beta,sigma_L = pars['alpha'],pars['beta'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for FIR_scaling_relation are "alpha","beta", "sigma_L" but {} were provided'.format(pars.keys()))
    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    L = np.zeros(len(SFR))*u.Lsun

    LIR_norm = LIR.to(u.erg/u.s)*(1/1e41)

    Lerg_norm = 10**(alpha*np.log10(LIR_norm.value)-beta)
    Lmean = (Lerg_norm*1e41*u.erg/u.s).to(u.Lsun)

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L[inds] = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)

    return L
