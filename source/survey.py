'''
Base module to make a LIM survey from painted lightcone
'''

import numpy as np
import dask.array as da
import astropy.units as u
import astropy.constants as cu
import copy
from nbodykit.source.catalog import ArrayCatalog

from source.lightcone import Lightcone
from source.utilities import cached_survey_property,get_default_params,check_params

class Survey(Lightcone):
    '''
    An object controlling all relevant quantities needed to create the 
    painted LIM lightcone. It reads a lightcone catalog of halos with SFR 
    quantities and paint it with as many lines as desired.
    
    Allows to compute summary statistics as power spectrum and the VID for 
    the signal (i.e., without including observational effects).
        
    INPUT PARAMETERS:
    ------------------
    
    -do_intensity           Bool, if True quantities are output in specific temperature
                            (Jy/sr units) rather than brightness temperature 
                            (muK units) 
                            (Default = False)
                    
    -Tsys:                  Instrument sensitivity. System temperature for brightness temperature
                            and noise equivalent intensitiy (NEI) for intensitiy (Default = 40 K)
    
    -Nfeeds:                Total number of feeds (detector*antennas*polarizations) (Default = 19)
        
    -nuObs_min,nuObs_max:   minimum and maximum ends of the frequency band (Default = 26-34 GHz)
    
    -RAObs_min,RAObs_max:   minimum and maximum RA observed (Default = -65-60 deg)
    
    -DECObs_min,DECObs_max: minimum and maximum DEC observed (Default = -1.25-1.25 deg)
    
    -dnu:                   Width of a single frequency channel (Default = 15.6 MHz)
    
    -beam_FWHM:             Beam full width at half maximum (Default = 4.1 arcmin)
    
    -tobs:                  Observing time on a single field (Default = 6000 hr)
                                
    -target_line:           Target line of the survey (Default: CO)
    
    -supersample:           Factor of supersample with respect to the survey resolution
                            when making the grid. (Default: 10)
    
    -paint_catalog:         Boolean: Paint catalog or used a painted one.               DOES THIS MAKE SENSE OR ALWAYS TRUE????
                            (Default: True). 
    
    -output_root            Root path for output products. (default: output/default)                                
    '''
    def __init__(self,
                 do_intensity=False,
                 Tsys_NEFD=40*u.K,
                 Nfeeds=19,
                 nuObs_min = 26.*u.GHz,
                 nuObs_max = 34.*u.GHz,
                 RAObs_min = -65.*u.deg,
                 RAObs_max = 60.*u.deg,
                 DECObs_min = -1.25*u.deg,
                 DECObs_max = 1.25*u.deg,
                 dnu=15.6*u.MHz,
                 beam_FWHM=4.1*u.arcmin,
                 tobs=6000*u.hr, 
                 target_line = 'CO',        
                 supersample = 10,         
                 output_root = "output/default",
                 paint_catalog = True,
                 **lightcone_kwargs):
                     
        # Initiate Lightcone() parameters
        Lightcone.__init__(self,**lightcone_kwargs)
        
        self._update_lightcone_list = self._update_lightcone_list
        
        self._survey_params = locals()
        self._survey_params.pop('self')
        self._survey_params.pop('lightcone_kwargs')
        self._default_survey_params = get_default_params(Survey.__init__)
        check_params(self._survey_params,self._default_survey_params)
        
        # Set survey parameters
        for key in self._survey_params:
            setattr(self,key,self._survey_params[key])
        
        # Combine lightcone_params with survey_params
        self._input_params.update(self._survey_params)
        self._default_params.update(self._default_survey_params)
        
        if self.paint_catalog:
            self.read_halo_catalog
            self.halo_luminosity
            
        #Set units for observable depending on convention
        if self.do_intensity:
            self.unit = u.Jy/u.sr
        else:
            self.unit = u.uK
            
        sigma_perp = 0.
        sigma_par = 0.

    @cached_survey_property
    def nuObs_mean(self):
        '''
        Mean observed frequency
        '''
        return 0.5*(self.nuObs_min+self.nuObs_max)
        
    @cached_survey_property
    def zmid(self):
        '''
        Effective mid redshift (obtained from nuObs_mean):
        '''
        return (self.line_nu0[self.target_line]/self.nuObs_mean).decompose()-1
                 
    @cached_survey_property
    def delta_nuObs(self):
        '''
        Experimental frequency bandwith
        '''
        return self.nuObs_max - self.nuObs_min
        
    @cached_survey_property
    def Omega_field(self):
        '''
        Solid angle covered by the survey
        '''
        return (self.RAObs_max-self.RAObs_min)*(self.DECObs_max-self.DECObs_min)
        
    @cached_survey_property
    def beam_width(self):
        '''
        Beam width defined as 1-sigma width of Gaussian beam profile
        '''
        return self.beam_FWHM*0.4247
        
    @cached_survey_property
    def Nside(self):
        '''
        Number of pixels per side of the observed map. RA,DEC
        '''
        return int(np.round(((self.RAObs_max-self.RAObs_min)/(self.beam_width)).decompose())),\
               int(np.round(((self.DECObs_max-self.DECObs_min)/(self.beam_width)).decompose()))
        
    @cached_survey_property
    def Npix(self):
        '''
        Number of pixels in the observed map
        '''
        return self.Nside[0]*self.Nside[1]
        
    @cached_survey_property
    def Nchan(self):
        '''
        Number of frequency channels in the observed map
        '''
        return int(np.round((self.delta_nuObs/(self.dnu)).decompose()))
        
    @cached_survey_property
    def sigmaN(self):
        '''
        Instrumental voxel noise standard deviation
        '''
        tpix = self.tobs/self.Npix
        if self.do_intensity:
            #intensity[Jy/sr]
            sig2 = self.Tsys**2/(self.Nfeeds*tpix)
        else:
            #Temperature[uK]
            sig2 = self.Tsys**2/(self.Nfeeds*self.dnu*tpix)
        return (sig2**0.5).to(self.unit)
        
    #########################
    ## Create the mock map ##
    #########################
        
    @cached_survey_property
    def halos_in_survey(self):
        '''
        Filters the halo catalog and only takes those that have observed
        frequencies within the experimental frequency bandwitdh and lie in the 
        observed RA - DEC ranges
        '''
        #empty catalog
        halos_survey = {}
        
        #halos within footprint
        inds_RA = (self.halo_catalog['RA'] > self.RAObs_min.value)&(self.halo_catalog['RA'] < self.RAObs_max.value)
        inds_DEC = (self.halo_catalog['DEC'] > self.DECObs_min.value)&(self.halo_catalog['DEC'] < self.DECObs_max.value)
        inds_sky = inds_RA&inds_DEC
        #Loop over lines to see what halos are within nuObs
        for line in self.lines.keys():
            if self.lines[line]:
                halos_survey[line] = dict(RA= np.array([]),DEC=np.array([]),Zobs=np.array([]),Ztrue=np.array([]),Lhalo=np.array([])*u.Lsun)
                inds = (self.nuObs_line_halo[line] >= self.nuObs_min)&(self.nuObs_line_halo[line] <= self.nuObs_max)&inds_sky
                halos_survey[line]['RA'] = np.append(halos_survey[line]['RA'],self.halo_catalog['RA'][inds])
                halos_survey[line]['DEC'] = np.append(halos_survey[line]['DEC'],self.halo_catalog['DEC'][inds])
                halos_survey[line]['Zobs'] = np.append(halos_survey[line]['Zobs'],(self.line_nu0[self.target_line]/self.nuObs_line_halo[line][inds]).decompose()-1)
                halos_survey[line]['Ztrue'] = np.append(halos_survey[line]['Ztrue'],self.halo_catalog['Z'][inds])
                halos_survey[line]['Lhalo'] = np.append(halos_survey[line]['Lhalo'],self.L_line_halo[line][inds])
                
        return halos_survey
        
    @cached_survey_property
    def obs_map(self):
        '''
        Generates the mock intensity map observed in Cartesian coordinates.
        
        Each line has their own Cartesian volume, zmid, and smoothing scales.
        Then, all contributions are added to the target volume
        '''
        maps = np.zeros([self.Nchan*self.supersample,self.Nside[0]*self.supersample,self.Nside[1]*self.supersample])
        #Loop over lines and add all contributions
        for line in self.lines.keys():
            if self.lines[line]:
                #Convert the halo position in each volume to Cartesian coordinates (from Nbodykit)
                ra,dec = da.broadcast_arrays(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'])
                ra,dec  = da.deg2rad(ra),da.deg2rad(dec)
                # cartesian coordinates
                x = da.cos(dec) * da.cos(ra)
                y = da.cos(dec) * da.sin(ra)
                z = da.sin(dec)
                pos = da.vstack([x,y,z]).T
                ra,dec,redshift = da.broadcast_arrays(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'],
                                                      self.halos_in_survey[line]['Ztrue'])
                #radial distances in Mpch/h
                distances = self.cosmo.comoving_radial_distance(z)*u.Mpc
                r = redshift.map_blocks(lambda z: (((self.cosmo.comoving_radial_distance(z)*u.Mpc).to(self.Mpch)).value), 
                                        dtype=redshift.dtype)
                cartesian_halopos = r[:,None] * pos
                #Locate the grid such that bottom left corner of the box is [0,0,0] which is the nbodykit convention.
                lategrid = np.array(cartesian_halopos.compute())
                for n in range(3):
                    if np.min(lategrid[:,n]) < 0:
                        lategrid[:,n] += np.abs(np.min(lategrid[:,n]))
                    else:
                        lategrid[:,n] -= np.min(lategrid[:,n])
                #Grid, voxel size and box size
                zmid = (self.line_nu0[line]/self.nuObs_mean).decompose().value-1
                global sigma_perp
                global sigma_par
                sigma_par = (cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
                sigma_perp = (self.cosmo.comoving_radial_distance(zmid)*u.Mpc*(self.beam_width/(1*u.rad))).to(self.Mpch).value
                Vcell = sigma_par*sigma_perp**2*self.Mpch**3
                Lbox = np.zeros(3)
                for i in range(3):
                    Lbox[i] = np.max(lategrid[:,i])-np.min(lategrid[:,i])
                Nmesh = np.array([self.supersample*self.Nchan.decompose(), 
                                  self.supersample*Lbox[1]/sigma_perp, 
                                  self.supersample*Lbox[2]/sigma_perp], dtype=int)
                #Compute the signal in each voxel
                Hubble = self.cosmo.hubble_parameter(self.halos_in_survey[line]['Ztrue'])*(u.km/u.Mpc/u.s)
                if self.do_intensity:
                    #intensity[Jy/sr]
                    signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo']/Vcell).to(self.unit)
                else:
                    #Temperature[uK]
                    signal = (cu.c**3*(1+self.halos_in_survey[line]['Ztrue'])**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo']/Vcell).to(self.unit)
                #Build Nbodykit catalog object
                nbodycat = np.empty(len(cartesian_halopos), dtype=[('Position', ('f8', 3)), ('Weight', 'f8')])
                nbodycat['Position'] = lategrid 
                nbodycat['Weight'] = signal.value 
                cat = ArrayCatalog(nbodycat, Nmesh=Nmesh, BoxSize=Lbox)
                #Convert to a mesh, weighting by signal
                mesh = cat.to_mesh(Nmesh=Nmesh, BoxSize=Lbox, weight='Weight',
                                   resampler='tsc',compensated=True)
                #Apply the filtering to smooth mesh
                mesh = mesh.apply(aniso_filter, mode='complex', kind='wavenumber')
                #paint the map and resample to [Nchannel,Npix^0.5,Npix^0.5] (and rescale by change in volume)
                maps += mesh.paint(mode='real')#,Nmesh = [self.Nchan,self.Nside,self.Nside])*Nmesh[1]*Nmesh[2]/self.Npix
        #Add the noise contribution 
        maps += np.random.normal(0.,self.sigmaN.value,maps.shape)
        
        return maps
                
                
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
def aniso_filter(k, v):
    '''
    Filter for k_perp and k_par modes separately.
    Applies to an nbodykit mesh object as a regular filter.

    Uses globally defined variables:
        sigma_perp - 'angular' smoothing in the flat sky approximation
        sigma_par - 'radial' smoothing from number of channels.

    Usage:
        mesh.apply(perp_filter, mode='complex', kind='wavenumber')

    NOTES:
    k[0] *= modifies the next iteration in the loop.
    Coordinates are fixed except for the k[1] which are
    the coordinate that sets what slab is being altered?

    '''
    #global sigma_perp
    #global sigma_par
    print(sigma_perp,sigma_par)
    rper = sigma_perp
    rpar = sigma_par
    newk = copy.deepcopy(k)


    #Smooth the k-modes anisotropically
    newk[0] *= rpar
    newk[1] *= rper
    newk[2] *= rper

    #Build smoothed values
    kk = sum(ki**2 for ki in newk)

    kk[kk==0]==1

    return np.exp(-0.5*kk)*v                
                    
