'''
Base module to make LIM measurements from a mock LIM survey from painted lightcone
'''

import numpy as np
import astropy.units as u
import astropy.constants as cu
from astropy.io import fits
import healpy as hp
from scipy.interpolate import interp2d,interp1d
from scipy.special import legendre
from nbodykit.algorithms import FFTPower
from nbodykit.source.mesh.array import ArrayMesh
from source.survey import Survey
from source.lightcone import Lightcone
from source.utilities import cached_measure_property,get_default_params,check_params
from source.utilities import set_lim, dict_lines

class Measure(Survey):
    '''
    Object containing all relevant attributes to compute summary statistics out 
    of the mock LIM survey prepared in the Survey class, created from painted
    lightcones. 
    
    Includes the 3d power spectrum Legendre multipoles, VID, angular statistics
    
    INPUT PARAMETERS:
    ------------------
    
    -dk:                    k spacing for the power spectrum (default: 0.02 Mpc^-1~0.01 h/Mpc)

    -kmin,kmax:             Minimum and maximum k values for the power spectrum
                            (default: 0., 3 Mpc^-1 ~ 5 h/Mpc)

    -Nmu:                   Number of sampling in mu to compute the power spectrum
                            (default: 10)

    -remove_noise:          Remove the expected instrumental noise power spectrum (sigma_N^2*Vvox)
                            from the observed power spectrum (and adds it to the covariance).
                            (default: False)
    
    -Tmin_VID,Tmax_VID:     Minimum and maximum values to compute the VID histogram
                            (default: 0.01 uK, 1000 uK)
                            
    -linear_VID_bin:        Boolean, to do linear (or log) binning for the VID histogram
                            (default: False)

    -Nbin_hist              Number of bins for the VID histogram
                            (default: 100)
                            
    -angular_map            Whether the map used is angular (healpy map). (Default: False)
    
    -do_read_map            Whether to read a map already saved from a survey computed before (Default:False)
    
    -map_name               The name of the map to read (Default: '')
    '''   
    def __init__(self,
                 dk = 0.02*u.Mpc**-1,
                 kmin = 0.0*u.Mpc**-1,
                 kmax = 3.*u.Mpc**-1,
                 Nmu = 5,
                 remove_noise = False,
                 Tmin_VID = 1.0e-2*u.uK,
                 Tmax_VID = 1000.*u.uK,
                 linear_VID_bin = False,
                 Nbin_hist = 100,
                 angular_map = False,
                 do_read_map = False,
                 map_name = '',
                 **lightcone_survey_kwargs):
                 
        # Initiate Survey() parameters
        Survey.__init__(self,**lightcone_survey_kwargs)
        
        self._update_lightcone_list = self._update_lightcone_list
        self._update_survey_list = self._update_survey_list

        self._measure_params = locals()
        self._measure_params.pop('self')
        self._measure_params.pop('lightcone_survey_kwargs')
        self._default_measure_params = get_default_params(Measure.__init__)
        check_params(self._measure_params,self._default_measure_params)
        
        # Set measure parameters
        for key in self._measure_params:
            setattr(self,key,self._measure_params[key])
            
        if self.do_angular != self.angular_map:
            raise ValueError("'do_angular' and 'angular_map' must be the same when the map is computed")
            
        # Combine measure_params with other classes
        self._input_params.update(self._measure_params)
        self._default_params.update(self._default_measure_params)
        
    ##################
    ## Read the map ##
    ##################
    
    @cached_measure_property
    def read_map(self):
        '''
        Reads a previously saved map to avoid rerun survey everytime
        '''
        if do_angular:
            mapread = hp.fitsfunc.read_map(self.map_name)
        else:
            #read the fits file
            hdul = fits.open(self.map_name)
            fitsmap = hdul[0].data
            hdul.close()
            #Transform it to a mesh field
            mesh = ArrayMesh(fitsmap.byteswap().newbyteorder(),BoxSize=self.Lbox.value)
            mapread = mesh.field
            
        return mapread
    
    ##########################################
    ## Fourier 3d power spectrum multipoles ##
    ##########################################
    
    @cached_measure_property
    def Pk_2d(self):
        '''
        Computes the 2d power spectrum P(k,mu) of the map
        '''
        if self.angular_map:
            print("ERROR!!! Fourier power spectrum measurements are available only if 'angular_map == False'")
            return None
        else:
            if not self.do_inner_cut:
                print("ERROR!!: Fourier power spectrum measurements are only available at the moment if 'do_inner_cut == True'")
                return None
            else:
                if self.do_read_map:
                    map_to_use = self.read_map
                else:
                    map_to_use = self.obs_3d_map
                    
                return FFTPower(map_to_use, '2d', Nmu=self.Nmu, poles=[0,2,4], los=[1,0,0],
                                dk=self.dk.to(self.Mpch**-1).value,kmin=self.kmin.to(self.Mpch**-1).value,
                                kmax=self.kmax.to(self.Mpch**-1).value,BoxSize=self.Lbox.value)

    @cached_measure_property
    def k_Pk_poles(self):
        '''
        Fourier wavenumbers for the multipoles of the power spectrum
        '''
        return (self.Pk_2d.poles['k']*self.Mpch**-1).to(self.Mpch**-1)

    @cached_measure_property
    def Pk_0(self):
        '''
        Monopole of the power spectrum
        '''
        if self.remove_noise:
            return (self.Pk_2d.poles['power_0'].real*self.Mpch**3).to(self.Mpch**3)*self.unit**2 - self.sigmaN**2*self.Vvox
        else:
            return (self.Pk_2d.poles['power_0'].real*self.Mpch**3).to(self.Mpch**3)*self.unit**2

    @cached_measure_property
    def Pk_2(self):
        '''
        Quadrupole of the power spectrum
        '''
        return (self.Pk_2d.poles['power_2'].real*self.Mpch**3).to(self.Mpch**3)*self.unit**2

    @cached_measure_property
    def Pk_4(self):
        '''
        Hexadecapole of the power spectrum
        '''
        return (self.Pk_2d.poles['power_4'].real*self.Mpch**3).to(self.Mpch**3)*self.unit**2
        
    ###################################################
    ## Theoretical Pk multipole covariance using lim ##
    ###################################################

    @cached_measure_property
    def Pk_2d_theo(self):
        '''
        Computes the anisotropic power spectrum from theory, using lim.
        Neglects potential cross-correlations between lines if volumes probed overlap.
        Returns k_obs,mu_obs,Pk
        '''
        #Call lim, and prepare it for the target line
        M = set_lim(self)
        line_model,line_pars = dict_lines(self,self.models[self.target_line]['model_name'],
                                          self.models[self.target_line]['model_pars'])
        if 'sigma_L' in self.models[self.target_line]['model_pars']:
            sigma_scatter = self.models[self.target_line]['model_pars']['sigma_L']
        else:
            sigma_scatter = 0.
        M.update(nu=self.line_nu0[self.target_line],model_name=line_model,model_par=line_pars,
                      sigma_scatter = sigma_scatter)
        M.update(Mmin = (np.min(self.halo_catalog['M_HALO'])*self.Msunh).to(u.Msun), Mmax = (np.max(self.halo_catalog['M_HALO'])*self.Msunh).to(u.Msun))
        M.update(sigma_NL=((np.trapz(M.PKint(M.z,M.k.value)*u.Mpc**3,M.k)/6./np.pi**2)**0.5).to(u.Mpc))

        PK_2d = M.Pk
        for line in self.lines.keys():
            if self.lines[line]:
                if line == self.target_line:
                    #already done above
                    continue
                #Repeat for all line interlopers
                line_model,line_pars = dict_lines(self,self.models[line]['model_name'],
                                          self.models[line]['model_pars'])
                M.update(nu=self.line_nu0[line],model_name=line_model,model_par=line_pars,
                              sigma_scatter = self.models[line]['model_pars']['sigma_L'])
                M.update(sigma_NL=((np.trapz(M.PKint(M.z,M.k.value)*u.Mpc**3,M.k)/6./np.pi**2)**0.5).to(u.Mpc))
                #Projection effects in the scales
                q_perp = M.cosmo.angular_diameter_distance([M.z])*(1+M.z)/(M.cosmo.angular_diameter_distance([self.zmid])*(1+self.zmid))
                q_par = (1.+M.z)/M.cosmo.hubble_parameter([M.z])/((1.+self.zmid)/M.cosmo.hubble_parameter([self.zmid]))
                F = q_par/q_perp
                prefac = 1./q_perp**2/q_par
                #Get "real" k and mu
                kprime = np.zeros((len(M.mu),len(M.k)))*M.k.unit
                mu_prime = M.mui_gridco/F/np.sqrt(1.+M.mui_grid**2.*(1./F/F-1))
                for imu in range(M.nmu):
                    kprime[imu,:] = M.k/q_perp*np.sqrt(1.+M.mu[imu]**2*(1./F/F-1))
                #Get the measured Pk contribution and add it to the rest, smooth it if necessary
                if self. smooth:
                    PK_2d += interp2d(M.k,M.mu,M.Pk*M.Wkmax)(kprime,mu_prime)*PK_2d.unit
                else:
                    PK_2d += interp2d(M.k,M.mu,M.Pk)(kprime,mu_prime)*PK_2d.unit

        return M.k.to(M.Mpch**-1),M.mu,PK_2d.to(self.Mpch**3*self.unit**2)

    @cached_measure_property
    def covmat_00(self):
        '''
        00 term of the total covariance matrix
        '''
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 0.5*np.trapz(integrand**2,self.Pk_2d_theo[1],axis=0)
        return (interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6).to(self.Mpch**6)*self.unit**4


    @cached_measure_property
    def covmat_02(self):
        '''
        02 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L2 = legendre(2)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 5./2.*np.trapz(integrand**2*L2,self.Pk_2d_theo[1],axis=0)
        return (interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6).to(self.Mpch**6)*self.unit**4


    @cached_measure_property
    def covmat_04(self):
        '''
        04 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L4 = legendre(4)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 9./2.*np.trapz(integrand**2*L4,self.Pk_2d_theo[1],axis=0)
        return (interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6).to(self.Mpch**6)*self.unit**4


    @cached_measure_property
    def covmat_22(self):
        '''
        22 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L2 = legendre(2)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 25./2.*np.trapz(integrand**2*L2*L2,self.Pk_2d_theo[1],axis=0)
        return (interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6).to(self.Mpch**6)*self.unit**4


    @cached_measure_property
    def covmat_24(self):
        '''
        24 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L2 = legendre(2)(mui_grid)
        L4 = legendre(4)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 45./2.*np.trapz(integrand**2*L2*L4,self.Pk_2d_theo[1],axis=0)
        return (interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6).to(self.Mpch**6)*self.unit**4


    @cached_measure_property
    def covmat_44(self):
        '''
        44 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L4 = legendre(4)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 81./2.*np.trapz(integrand**2*L4*L4,self.Pk_2d_theo[1],axis=0)
        return (interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6).to(self.Mpch**6)*self.unit**4


    def get_covmat(self,Nmul):
        '''
        Get the covariance matrix for a given number of multipoles
        (starting always from the monopole and without skipping any pair
        multipole)
        '''
        if Nmul > 3:
            raise ValueError('Not implemented yet!\
            Implement covmat_66 and expand this function')

        nk = len(self.k_Pk_poles)
        covmat = np.zeros((nk*Nmul,nk*Nmul))*self.covmat_00.unit
        covmat[:nk,:nk] = np.diag(self.covmat_00)

        if Nmul > 1:
            covmat[:nk,nk:nk*2] = np.diag(self.covmat_02)
            covmat[nk:nk*2,:nk] = np.diag(self.covmat_02)
            covmat[nk:nk*2,nk:nk*2] = np.diag(self.covmat_22)
            covmat[:nk,nk:nk*2] = np.diag(self.covmat_02)
        if Nmul > 2:
            covmat[:nk,nk*2:nk*3] = np.diag(self.covmat_04)
            covmat[nk:nk*2,nk*2:nk*3] = np.diag(self.covmat_24)
            covmat[nk*2:nk*3,:nk] = np.diag(self.covmat_04)
            covmat[nk*2:nk*3,nk:nk*2] = np.diag(self.covmat_24)
            covmat[nk*2:nk*3,nk*2:nk*3] = np.diag(self.covmat_44)

        return covmat
        
    ######################
    ## VID measurements ##
    ######################
    
    @cached_measure_property
    def Ti_edge(self):
        '''
        Edges of the VID histogram bins
        '''
        if self.linear_VID_bin:
            Te = np.linspace(self.Tmin_VID.value,self.Tmax_VID.value,self.Nbin_hist+1)*self.Tmin_VID.unit
        else:
            Te = np.logspace(np.log10(self.Tmin_VID.value),np.log10(self.Tmax_VID.value),self.Nbin_hist+1)*self.Tmin_VID.unit
        return Te

    @cached_measure_property
    def Ti(self):
        '''
        Center of the VID histogram bins
        '''
        return (self.Ti_edge[:-1]+self.Ti_edge[1:])/2.

    @cached_measure_property
    def Bi_VID(self):
        '''
        Computes the histogram of temperatures in each voxel in hte observed map.
        Equivalent to the VID
        '''
        return np.histogram(np.array(self.obs_3d_map).flatten(),
                            bins=self.Ti_edge.value)[0]

    @cached_measure_property
    def Bi_VID_covariance(self):
        '''
        Covariance matrix of the VID histograms
        '''
        return np.diag(self.Bi_VID)
