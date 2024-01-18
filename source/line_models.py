'''
Catalog of different lines and models
'''

import numpy as np
from numpy.random import normal,multivariate_normal
from scipy.special import gamma, gammainc
from scipy.interpolate import interp1d, RegularGridInterpolator
import astropy.units as u
import astropy.constants as cu
import gc
from source.utilities import newton_root
from time import time
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

def LIR(self,SFR,Mstar,pars,rng):
    '''
    Obtain the IR luminosity from SFR or stellar mass

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -Mstar:     Stellar mass of the halo in Msun
        -pars:      Dictionary of parameters
            -IRX_name:      The reference to use the IRX from
            -IRX_params:    The parameters required to compute the IRX
                            (check each if instance below)
            -K_IR, K_UV:    The coefficients to relate SFR to L_IR and L_UV
        -rng:       RNG object with the seed set in the input
    '''
    #avoid SFR=0 issues
    LIR = np.zeros_like(SFR)*u.Lsun
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
            IRX = 10**log10IRX_0*Mstar[inds]
        #IRX - Mstar relation from Bouwens 2020, arXiv:2009.10727
        elif pars['IRX_name'] == 'Bouwens2020':
            log10Ms_IRX,alpha_IRX,sigma_IRX = pars['log10Ms_IRX'],pars['alpha_IRX'],pars['sigma_IRX']
            IRX = (Mstar[inds]/10**log10Ms_IRX)**alpha_IRX
        #IRX - Mstar relation from Heinis 2014, arXiv:1310.3227
        elif pars['IRX_name'] == 'Heinis2014':
            log10Ms_IRX,alpha_IRX,log10IRX_0,sigma_IRX = pars['log10Ms_IRX'],pars['alpha_IRX'],pars['log10IRX_0'],pars['sigma_IRX']
            IRX = (Mstar[inds]/10**log10Ms_IRX)**alpha_IRX*10**log10IRX_0
        else:
            raise ValueError('Please choose a valid IRX model')
        #Add mean-preserving lognormal scatter in the IRX relation
        sigma_base_e = sigma_IRX*2.302585
        IRX = IRX*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, IRX.shape)

        K_IR,K_UV = pars['K_IR'],pars['K_UV']

    LIR[inds] = SFR[inds]/(K_IR + K_UV/IRX)*u.Lsun

    return LIR
    

def LIR_and_LUV(self,SFR,Mstar,pars,rng):
    '''
    Obtain the IR and UV luminosities from SFR or stellar mass
    
    -to use outside code

    Parameters:
        -SFR:       SFR of the halo in Msun/yr
        -Mstar:     Stellar mass of the halo in Msun
        -pars:      Dictionary of parameters
            -IRX_name:      The reference to use the IRX from
            -IRX_params:    The parameters required to compute the IRX
                            (check each if instance below)
            -K_IR, K_UV:    The coefficients to relate SFR to L_IR and L_UV
        -rng:       RNG object with the seed set in the input
    '''
    #avoid SFR=0 issues
    LIR = np.zeros_like(SFR)*u.Lsun
    LUV = np.zeros_like(SFR)*u.Lsun
    inds = np.where(SFR>0)[0]

    #Try to get IRX:
    if 'IRX_name' not in pars:
        raise ValueError('To use this function you need to choose an IRX model')
    else:
        #IRX - Mstar relation from Bouwens 2016, arXiv:1606.05280
        if pars['IRX_name'] == 'Bouwens2016':
            log10IRX_0,sigma_IRX = pars['log10IRX_0'],pars['sigma_IRX']
            IRX = 10**log10IRX_0*Mstar[inds]
        #IRX - Mstar relation from Bouwens 2020, arXiv:2009.10727
        elif pars['IRX_name'] == 'Bouwens2020':
            log10Ms_IRX,alpha_IRX,sigma_IRX = pars['log10Ms_IRX'],pars['alpha_IRX'],pars['sigma_IRX']
            IRX = (Mstar[inds]/10**log10Ms_IRX)**alpha_IRX
        #IRX - Mstar relation from Heinis 2014, arXiv:1310.3227
        elif pars['IRX_name'] == 'Heinis2014':
            log10Ms_IRX,alpha_IRX,log10IRX_0,sigma_IRX = pars['log10Ms_IRX'],pars['alpha_IRX'],pars['log10IRX_0'],pars['sigma_IRX']
            IRX = (Mstar[inds]/10**log10Ms_IRX)**alpha_IRX*10**log10IRX_0
        else:
            raise ValueError('Please choose a valid IRX model')
        #Add mean-preserving lognormal scatter in the IRX relation
        sigma_base_e = sigma_IRX*2.302585
        IRX = IRX*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, IRX.shape)

        K_IR,K_UV = pars['K_IR'],pars['K_UV']

    LIR[inds] = SFR[inds]/(K_IR + K_UV/IRX)*u.Lsun
    LUV[inds] = LIR[inds]/IRX

    return LIR,LUV

##########################
## Cosmic IR Background ##
##########################

def CIB_band_Agora(self,halos,LIR,pars,itau_nu0,tau_nu0_norm):
    '''
    Model for the CIB luminosity in a given observed band.
    Follows the modeling implemented in the Agora simulations
    (arXiv:2212.07420)

    Parameters:
        -halos:     Halos to take into account (with all halo props)
        -SFR:       SFR of the halo in Msun/yr
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model (emulator parameters)
            -B:         parameter of the gas-to-stellar mass ratio
            -zeta_d:     
            -A_d:       Normalization of the dust mass to Tdust relation 
            -alpha:     Power of the suppression in Mdust/Mstar at z>2
    '''
    try:
        B,zeta_d,A_d,alpha = pars['B'],pars['zeta_d'],pars['A_d'],pars['alpha']
    except:
        raise ValueError('The model_pars for CIB_Agora are "B", "zeta_d", "A_d", and "alpha", but {} were provided'.format(pars.keys()))

    #We consider only galaxies that are forming stars and have stellar mass
    Mstar = halos['Mstar']
    SFR = halos['SFR']
    inds = (SFR>0)&(Mstar>0)
    hidx = np.argwhere(inds)
    #Get the dust temperature and gray body parameter for all halos
    Tdust = Tdust_Agora(halos['Zobs'][inds],SFR[inds],Mstar[inds],LIR[inds],
                        B,zeta_d,A_d,alpha)
    #rest frame frequency for each halo corresponding to the observed bandwidth
    nu0 = np.geomspace(self.nuObs_min.value,self.nuObs_max.value,self.NnuObs)*self.nuObs_min.unit

    #Read the imaging band table
    tau_nu0 = itau_nu0(nu0)

    #Prepare to compute the color correction term
    if self.nu_c == None:
        nu_c = np.trapz(nu0*tau_nu0,nu0)/tau_nu0_norm
    else:
        nu_c = self.nu_c
    Cc_norm = np.trapz(nu_c/nu0*tau_nu0,nu0)
   
    L_CIB = np.zeros(len(LIR))
    #Hard coded right now
    Niter = 100
    nsubcat = len(halos[inds])//Niter 

    #Generate lookup tables for IR SEDs as a function of temperature
    SEDSpl = SEDTabulate(self,nu0)
    NormSpl = make_SEDnorm(self) 
    #Iteratively compute the SED contribution to L_CIB for each halo
    for i in range(Niter+1):
        if i == Niter:
            #last iteration
            Td = Tdust[i*nsubcat:].value
            zhalo = halos[inds]['Zobs'][i*nsubcat:]
        else:
            Td = Tdust[i*nsubcat:(i+1)*nsubcat].value 
            zhalo = halos[inds]['Zobs'][i*nsubcat:(i+1)*nsubcat]
        #Quick evaluation of full SED on nu0 for each entry
        SEDhalos = SEDSpl((zhalo, Td))
        SEDnorm = NormSpl(Td)

        iSEDSpl = interp1d(nu0,SEDhalos)(nu_c)
        Cc = np.trapz(tau_nu0*SEDhalos/iSEDSpl[:,None],nu0)/Cc_norm #Color correction

        #(1+zhalo)^2 added in the integralto do the integral in restframe nu and to set tau in rest-frame nu
        int_term = Cc*np.trapz(SEDhalos*tau_nu0[None,:]*(1+zhalo[:,None])**2, nu0.value, axis=1)/SEDnorm/tau_nu0_norm * self.rng.normal(1, 0.25, len(SEDnorm))    
        if i== Niter:
            L_CIB[hidx[i*nsubcat:][:,0]] = int_term
        else:
            L_CIB[hidx[i*nsubcat:(i+1)*nsubcat][:,0]] = int_term
    #Get the SED normalization
    #Compute the L_CIB that each halo contributes to the band (giving SED its unit)
    #The units are super funky right now.. Need to double check NOTE -> JOSE: Units are Lsun (from LIR) / GHz (from the SED). tau_nu0 has no units
    #L_CIB[inds] = LIR[inds]*L_CIB[inds]
    #return L_CIB
    return LIR*L_CIB/u.GHz

def make_SEDnorm(self):
    '''
    Generate a function which computes the normalization of the SED
    for a dust temperature Td [K]
    '''
    nus = np.geomspace(0.1, 1e7, 1000)*u.GHz
    tablespl = SEDTabulate(self,nus)
    Tdvec = np.geomspace(0.05, 100, 2000)
    
    #Generate table with broad range in nus, same values of T as other function, compute normalization
    tableSED = tablespl((self.zmin,Tdvec))
    #print(tableSED.shape)
    #print(nus.shape)
    norm = np.trapz(tableSED, nus, axis=1)*(1+self.zmin) #correction factor to do this integral at same frame
    normspl = interp1d(Tdvec, norm)
    return normspl

def SEDTabulate(self,nu):
    '''
    Given an input set of frequencies nu, generate SED for Td. 
    '''

    Tdvec = np.geomspace(0.05, 100, 2000)
    dz = 0.01
    zvec = np.arange(self.zmin-dz,self.zmax+dz,dz)
    
    #Add units
    Td = Tdvec*u.K
    alpha_d=2
    betad = 1.25 / (0.4 + 0.008*Tdvec)

    #Build the SEDvec
    SEDvec = np.zeros(shape=(len(nu), len(zvec), len(betad)))

    kTh = (cu.k_B * Td / (cu.h)).to(u.GHz)    
    nup = kTh*(3+betad+alpha_d)

    #Build the mask
    bignuvec = np.tile(nu, len(betad)*len(zvec)).reshape(len(betad), len(zvec), len(nu)).T
    #Scale each of the frequencies coordinates to the rest frame frequencies at z
    bignuvec = np.einsum('nzT, z->nzT', bignuvec, (1+zvec))

    #Compute nu < nup
    nu_inds = bignuvec <= nup
    SEDvec[nu_inds] = np.exp((np.einsum('nzh, h->nzh', np.log(bignuvec.value),betad+3)) - np.log(np.exp(np.einsum('nzh, h->nzh', bignuvec, 1./kTh)) - 1)) [nu_inds]
    #Compute nu > nup
    nu_inds = ~nu_inds
    SEDvec[nu_inds] = np.einsum('nzh, h->nzh', bignuvec**-alpha_d, nup.value**(betad+3+alpha_d)/(np.exp(nup/kTh) - 1))[nu_inds]
    SEDspl = RegularGridInterpolator((zvec, Td), np.einsum('nzh->zhn', SEDvec))

    return SEDspl

def Tdust_Agora(z,SFR,Mstar,LIR,B,zeta_d,A_d,alpha):
    '''
    Computes the dust temperature within a halo following the implementation in 
    the Agora simulations (arXiv:2212.07420) 

    Depends on redshift, metalicity, Mgas to Mstar ratio, 
    specific star-formation rate

    Returns both Tdust and the index beta_d for the SED for each halo
    '''
    #Gas metallicity from the empirical relation from Sanders et al 2021
    #   (arXiv:2009.07292)
    st = time()
    y = np.log10(Mstar) - 0.6 * np.log10(SFR) - 10
    Zgas = 8.80 + 0.188*y - 0.220 * y**2 - 0.0531 * y**3
    del y
    gc.collect()
    #print(time() - st, "y and zgas")
    #Main-sequence specific star-formation state from Tacconi et al 2018,
    #   (arXiv:1702.01140), using their "cosmic-time(z)" fit formula
    #   Note there is a difference of 10^9 because our sSFR is in Yr^-1 and
    #   theirs are in Gyr^-1.
    logz = np.log10(1+z)
    tc = 10**(1.143 - 1.026*logz - 0.599*logz**2 + 0.528*logz**3)
    sSFR_MS = 10**((-0.16-0.026*tc) * (np.log10(Mstar)+0.025) - (6.51-0.11*tc))
    del tc
    gc.collect()
    #print(time() - st, "tc and logz")
    #Gas-to-stellar mass ratio following Tacconi et al 2020, arXiv:2003.06245
    A,C,D,F = 0.06,0.51,-0.41,0.65
    sSFR = SFR/Mstar
    Mgas_over_Mstar = 10**(A + B*(logz-F)**2 + C*np.log10(sSFR/sSFR_MS) + D*(np.log10(Mstar)-10.7))
    del sSFR, sSFR_MS, logz
    gc.collect()
    #print(time() - st, "Mgas Mstar")
    #Compute suppression in the Mdust-to-Mstar ratio at z>2, from 
    #   Donevski et al. 2020 (arXiv:2008.09995)
    factor = np.ones_like(z)
    zinds = z > 2
    factor[zinds] = (1-0.05*z[zinds])**alpha
    #Unnormalized dust mass to stellar mass ratio from the relation in 
    #   Donevski et al. 2020 (arXiv:2008.09995)
    Mdust = Mstar * Mgas_over_Mstar * Zgas * factor
    del factor
    gc.collect()
    #print(time() - st, "Mdust get")
    #Find dust temperature and the index of its relation with IR luminosity
    beta_d = newton_root(beta_d_function,beta_d_derivative,2.5,LIR.value,Mdust,zeta_d,A_d,Niter=5)
    #print(time() - st, "Root find Niter=5")
    Tdust = A_d * (LIR.value/Mdust)**(1/(4+beta_d))
    return Tdust*u.K#, beta_d

def beta_d_function(beta_d,LIR,Mdust,zeta_d,A_d):
    '''
    Function to find the root (i.e., beta_d)
    '''
    P3 = LIR/Mdust
    P1,P2 = 125*zeta_d,125*0.4
    return P1/beta_d - P2 - A_d*P3**(1/(4+beta_d))

def beta_d_derivative(beta_d,LIR,Mdust,zeta_d,A_d):
    '''
    Derivative to find the root (i.e., beta_d)
    '''
    P3 = LIR/Mdust
    P1 = 125*zeta_d
    return A_d*P3**(1/(4+beta_d))*np.log(P3)/(beta_d+4)**2-P1/beta_d**2

##############
## CO LINES ##
##############

def CO_Li16(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Model for CO(1-0) line from Li+2016 (arXiv:1503.08833)

    Parameters:
        -halos:     Halos to take into account (with all halo props)
        -SFR:       SFR of the halo in Msun/yr
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model
            -delta_mf:  IMF normalization
            -alpha:     power law coefficient relating IR and CO luminosities
            -beta:      Multiplicative normalization for the IR and CO luminosities
            -sigma_L: Scatter in dex of the CO luminosity
        -rng:       RNG object with the seed set in the input
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
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model
            -alpha
            -beta
            -sigma_L    Scatter in dex of the luminosity
        -rng:       RNG object with the seed set in the input
    '''
    try:
        alpha,beta,sigma_L = pars['alpha'],pars['beta'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for CO_lines_scaling_LFIR are "alpha","beta", "sigma_L" but {} were provided'.format(pars.keys()))

    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    L = np.zeros(len(SFR))*u.Lsun

    Lp = 10**((np.log10(LIR[inds].value)-beta)/alpha)

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
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model
            -aLCII,bLCII    Fit to log10(L_CII/Lsun) = aLCII*log10(SFR/(Msun/yr)) + bLCII
            -sigma_L: Scatter in dex of the CII luminosity
        -rng:       RNG object with the seed set in the input
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
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model
            -alpha1, alpha2, beta1, beta2    Fit to log10(L_CII/Lsun) = alpha*log10(SFR/(Msun/yr)) + beta, where alpha=alpha1 + alpha2*z and beta=beta1 + beta2*z
            -sigma_L: Scatter in dex of the CII luminosity
        -rng:       RNG object with the seed set in the input
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
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model
            -C      Conversion between SFR and Ly-alpha luminosity
            -xi, zeta, psi, z0, f0    Parametrize the escape fraction, reflecting the possibility of photons being absorbed by dust
            -sigma_L    log-normal scatter in the Ly-alpha luminosity
        -rng:       RNG object with the seed set in the input
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
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model
            -M0, Mmin, alpha    Normalization, cutoff mass, and slope in the M_HI-M_halo relation
            -sigma_L    log-normal scatter in the luminosity
        -rng:       RNG object with the seed set in the input
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
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model
            -K            linear factor SFR = K*L (L in ergios/s)
            -Aext         Extinction of the line
            -sigma_L: Scatter in dex of the luminosity
        -rng:       RNG object with the seed set in the input
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
        -LIR:       Infrared luminosity in Lsun
        -pars:      Dictionary of parameters for the model
            -alpha
            -beta
            -sigma_L    Scatter in dex of the luminosity
        -rng:       RNG object with the seed set in the input
    '''
    try:
        alpha,beta,sigma_L = pars['alpha'],pars['beta'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for FIR_scaling_relation are "alpha","beta", "sigma_L" but {} were provided'.format(pars.keys()))
    #avoid SFR=0 issues
    inds = np.where(SFR>0)
    L = np.zeros(len(SFR))*u.Lsun

    LIR_norm = LIR[inds].to(u.erg/u.s)*(1/1e41)

    Lerg_norm = 10**(alpha*np.log10(LIR_norm.value)-beta)
    Lmean = (Lerg_norm*1e41*u.erg/u.s).to(u.Lsun)

    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L[inds] = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)

    return L

###################################
## General double power-law L(M) ##
###################################

def LofM_DoublePower(self,halos,SFR,LIR,pars,nu0,rng):
    '''
    Returns the luminosity for any line according to a double power law following the parametrization
    L/Lsun (M) = C/((M/M_\star)^A + (M/M_\star)^B) adding a scatter of sigma_L dex.
    From e.g., Chung et al 2022 (2111.05931)
    '''
    try:
        A, B, C, Mstar,sigma_L = pars['A'],pars['B'],pars['C'],pars['Mstar'],pars['sigma_L']
    except:
        raise ValueError('The model_pars for LofM_DoublePower are A, B, C, Mstar but {} were provided'.format(pars.keys()))
        
    #Get the halo masses in Msun
    Mhalo_Msun = (halos['M_HALO']*self.Msunh).to(u.Msun)
    ratio = Mhalo_Msun/Mstar

    Lmean = C/(ratio**A + ratio**B)
    #Add scatter to the relation
    sigma_base_e = sigma_L*2.302585
    L = Lmean*rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, Lmean.shape)
    return L
