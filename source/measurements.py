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
from nbodykit.source.mesh.catalog import CompensateCICShotnoise,get_compensation
from source.survey import Survey
from source.lightcone import Lightcone
from source.utilities import cached_measure_property,get_default_params,check_params,CompensateNGPShotnoise

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
                            
    -lmax:                  Maximum multipole to compute the angular power spectrum. Default=1000

    -remove_noise:          Remove the expected instrumental noise power spectrum (sigma_N^2*Vvox)
                            from the observed power spectrum (and adds it to the covariance).
                            (default: False)
                            
    -angular_map            Whether the map used is angular (healpy map). (Default: False)
    
    -do_read_map            Whether to read a map already saved from a survey computed before (Default:False)
    
    -map_name               The name of the map to read (Default: '')
    '''   
    def __init__(self,
                 dk = None,
                 kmin = 0.0*u.Mpc**-1,
                 kmax = None,
                 Nmu = 5,
                 lmax = 1000,
                 remove_noise = False,
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
        check_params(self,self._measure_params,self._default_measure_params)
        
        # Set measure parameters
        for key in self._measure_params:
            setattr(self,key,self._measure_params[key])
            
        if self.do_angular != self.angular_map:
            raise ValueError("'do_angular' and 'angular_map' must be the same when the map is computed")
            
        # Combine measure_params with other classes
        self._input_params.update(self._measure_params)
        self._default_params.update(self._default_measure_params)
        # Load in compensation function from the inherited lightcone survey resampler
        if self.resampler=='nearest':
            #Jing et al equation 20 states that the `CompensateNGPShotnoise` function is just 1. 
            self.compensation = [('Complex', CompensateNGPShotnoise, "circular")]
        else:
            #We're not doing interlacing so get the approximate correction instead
            self.compensation = get_compensation(interlaced=False,resampler=self.resampler)
            
            
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
            if self.cube_mode == 'outer_cube':
                print("ERROR!!: Fourier power spectrum measurements for 'cube_mode = outer_cube' are not available")
                return None
            else:
                if self.do_read_map:
                    map_to_use = self.read_map
                else:
                    map_to_use = self.obs_3d_map
                    
                #Compensate the field for the CIC window function we apply
                map_to_use = (map_to_use.r2c().apply(self.compensation, kind='circular')).c2r()
                
                try:
                    dk = self.dk.to(self.Mpch**-1).value
                except:
                    dk = self.dk
                    
                try:
                    kmax = self.kmax.to(self.Mpch**-1).value
                except:
                    kmax = self.kmax
                    
                return FFTPower(map_to_use, '2d', Nmu=self.Nmu, poles=[0,2], los=[1,0,0],
                                dk=dk,kmin=self.kmin.to(self.Mpch**-1).value,
                                kmax=kmax,BoxSize=self.Lbox.value)

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

    def c_ell(self):
        '''
        Computes the angular power spectrum C_ell of the map
        uses the fsky approximation to deconvolve mask effects. 

        TO-DO: Could switch to namaster implementation.
        '''
        map_to_use = self.obs_2d_map
        fsky = (self.RAObs_max - self.RAObs_min).to(u.radian)*(self.DECObs_max - self.DECObs_min).to(u.radian) / (4*np.pi * u.sr)
        cl = hp.anafast(map_to_use, lmax=self.lmax)/fsky

        print('With fsky=%0.3f your ell_min should be around %d so be careful above that'%(fsky, 1./fsky))

        return cl
