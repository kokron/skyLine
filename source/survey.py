'''
Base module to make a LIM survey from painted lightcone
'''

import numpy as np
import dask.array as da
import astropy.units as u
import astropy.constants as cu
import copy
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.algorithms import FFTPower
import pmesh
from pmesh.pm import RealField, ComplexField
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
                            
    -Tmin_VID,Tmax_VID:     Minimum and maximum values to compute the VID histogram
                            (default: 0.01 uK, 1000 uK)
                            
    -Nbin_hist              Number of bins for the VID histogram
                            (default: 100)
    
    -supersample:           Factor of supersample with respect to the survey resolution
                            when making the grid. (Default: 10)     
                                                   
    -dk:                    k spacing for the power spectrum (default: 0.003 Mpc^-1~0.005 h/Mpc)
    
    -kmin:                  Minimum k value for the power spectrum (default: 0.007 Mpc^-1~0.01 h/Mpc)
                                                
    -Nmu:                   Number of sampling in mu to compute the power spectrum
                            (default: 10)
                            
    -linear_VID_bin:        Boolean, to do linear (or log) binning for the VID histogram
                            (default: False)
    
    -paint_catalog:         Boolean: Paint catalog or used a painted one.               DOES THIS MAKE SENSE OR ALWAYS TRUE????
                            (Default: True). 
    
    -output_root            Root path for output products. (default: output/default)                                
    '''
    def __init__(self,
                 do_intensity=False,
                 Tsys=40*u.K,
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
                 Tmin_VID = 1.0e-2*u.uK,
                 Tmax_VID = 1000.*u.uK,
                 Nbin_hist = 100,  
                 linear_VID_bin = False,  
                 supersample = 10,     
                 dk = 0.003*u.Mpc**-1,
                 kmin = 0.007*u.Mpc**-1,
                 Nmu = 10,    
                 output_root = "output/default",
                 paint_catalog = True,
                 pmeshpaint = True,
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
        
        # Check that the observed footprint is contained in the lightcone
        if self.RAObs_min < self.RA_min or self.RAObs_max > self.RA_max or \
           self.DECObs_min < self.DEC_min or self.DECObs_max > self.DEC_max:
               raise ValueError('Please, your observed limits RA_Obs=[{},{}], DEC_Obs=[{},{}] must be within the lightcone limits RA=[{},{}], DEC=[{},{}].'.format(self.RAObs_min,self.RAObs_max,self.DECObs_min,self.DECObs_max,self.RA_min,self.RA_max,self.DEC_min,self.DEC_max))
        
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
    def obs_fourier_map(self):
        '''
        Generates the mock intensity map observed in Fourier space,
        obtained from Cartesian coordinates.
        
        Use obs_fourier_map.c2r() to get the real field
        '''
        #Define the mesh divisions
        Nmesh = np.array([self.supersample*self.Nchan, 
                  self.supersample*self.Nside[0], 
                  self.supersample*self.Nside[1]], dtype=int)
        #box angular limits
        RAlims = np.array([self.RAObs_min.value,self.RAObs_max.value])
        DEClims = np.array([self.DECObs_min.value,self.DECObs_max.value])
        ra,dec  = np.deg2rad(RAlims),np.deg2rad(DEClims)
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        pos_lims = np.vstack([x,y,z]).T
        #box size in observed redshift
        Zlims = (self.line_nu0[self.target_line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
        r = ((self.cosmo.comoving_radial_distance(Zlims)*u.Mpc).to(self.Mpch)).value
        grid_lim = r*pos_lims
        Lbox = np.zeros(3)
        for i in range(3):
            Lbox[i] = np.max(grid_lim[:,i])-np.min(lategrid[:,i])
        #Loop over lines and add all contributions        
        global sigma_par
        global sigma_perp
        maps = np.zeros([Nmesh[0],Nmesh[1],Nmes[2]//2 + 1],dtype='complex64')
        for line in self.lines.keys():
            if self.lines[line]:
                #Get true cell volume
                Zlims = (self.line_nu0[lines].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
                r = ((self.cosmo.comoving_radial_distance(Zlims)*u.Mpc).to(self.Mpch)).value
                grid_lim = r*pos_lims
                Lbox_true = np.zeros(3)
                for i in range(3):
                    Lbox_true[i] = np.max(grid_lim[:,i])-np.min(lategrid[:,i])
                Vcell_true = Lbox_true[0]*Lbox_true[1]*Lbox_true[2]/(Nmesh[0]*Nmesh[1]*Nmesh[2])*(self.Mpch**3).to(self.Mpch**3)
                #Get positions using the observed redshift
                #Convert the halo position in each volume to Cartesian coordinates (from Nbodykit)
                ra,dec = da.broadcast_arrays(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'])
                ra,dec  = da.deg2rad(ra),da.deg2rad(dec)
                # cartesian coordinates
                x = da.cos(dec) * da.cos(ra)
                y = da.cos(dec) * da.sin(ra)
                z = da.sin(dec)
                pos = da.vstack([x,y,z]).T
                ra,dec,redshift = da.broadcast_arrays(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'],
                                                      self.halos_in_survey[line]['Zobs'])
                #radial distances in Mpch/h
                r = redshift.map_blocks(lambda zz: (((self.cosmo.comoving_radial_distance(zz)*u.Mpc).to(self.Mpch)).value), 
                                        dtype=redshift.dtype)
                cartesian_halopos = r[:,None] * pos
                #Locate the grid such that bottom left corner of the box is [0,0,0] which is the nbodykit convention.
                lategrid = np.array(cartesian_halopos.compute())
                for n in range(3):
                    if np.min(lategrid[:,n]) < 0:
                        lategrid[:,n] += np.abs(np.min(lategrid[:,n]))
                    else:
                        lategrid[:,n] -= np.min(lategrid[:,n])
                #Compute the signal in each voxel (with Ztrue and Vcell_true)
                Hubble = self.cosmo.hubble_parameter(self.halos_in_survey[line]['Ztrue'])*(u.km/u.Mpc/u.s)
                if self.do_intensity:
                    #intensity[Jy/sr]
                    signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                else:
                    #Temperature[uK]
                    signal = (cu.c**3*(1+self.halos_in_survey[line]['Ztrue'])**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                #compute scales for the anisotropic filter (in Ztrue -> zmid)
                zmid = (self.line_nu0[line]/self.nuObs_mean).decompose().value-1
                sigma_par = (cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
                sigma_perp = (self.cosmo.comoving_radial_distance(zmid)*u.Mpc*(self.beam_width/(1*u.rad))).to(self.Mpch).value
                #Set the emitter in the grid and paint using pmesh directly instead of nbk
                pm = pmesh.pm.ParticleMesh(Nmesh, BoxSize=Lbox, dtype='float32', resampler='tsc')
                #Make realfield object
                field = pm.create(type='real')
                layout = pm.decompose(lategrid)
                #Exchange positions between different MPI ranks 
                p = layout.exchange(lategrid)
                #Assign weights following the layout of particles
                m = layout.exchange(signal.value)
                pm.paint(p, out=field, mass=m, resampler='tsc')
                #Add noise in the cosmic volume probed by target line
                if line == self.target_line:
                    #distribution is positive gaussian with 0 mean
                    vec = np.linspace(0.,6*self.sigmaN,1024)
                    exparg = -0.5*(vec/self.sigmaN)**2.
                    PDF = np.exp(exparg)
                    PDF *= 1./(np.sum(PDF))
                    field += np.random.choice(vec,field.shape,p=PDF)
                #Fourier transform fields and apply the filter
                field = field.r2c()
                field = field.apply(aniso_filter, kind='wavenumber')
                #Add this contribution to the total maps
                maps+=field
        return maps
                    
                
    
    
    
    
    @cached_survey_property
    def obs_map(self):
        '''
        Generates the mock intensity map observed in Cartesian coordinates.
        
        Each line has their own Cartesian volume, zmid, and smoothing scales.
        Then, all contributions are added to the target volume
        '''

        if self.pmeshpaint:
            maps = np.zeros([self.Nchan*self.supersample,self.Nside[0]*self.supersample,self.Nside[1]*self.supersample//2 + 1],dtype='complex64')
        else:

            maps = np.zeros([self.Nchan*self.supersample,self.Nside[0]*self.supersample,self.Nside[1]*self.supersample],dtype='float32')
        #Loop over lines and add all contributions
        global sigma_par
        global sigma_perp
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
                Lbox = np.zeros(3)
                for i in range(3):
                    Lbox[i] = np.max(lategrid[:,i])-np.min(lategrid[:,i])
                Nmesh = np.array([self.supersample*self.Nchan, 
                                  self.supersample*self.Nside[0], 
                                  self.supersample*self.Nside[1]], dtype=int)
                Vcell = Lbox[0]*Lbox[1]*Lbox[2]/(Nmesh[0]*Nmesh[1]*Nmesh[2])*self.Mpch**3
                #Compute the signal in each voxel
                Hubble = self.cosmo.hubble_parameter(self.halos_in_survey[line]['Ztrue'])*(u.km/u.Mpc/u.s)
                if self.do_intensity:
                    #intensity[Jy/sr]
                    signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo']/Vcell).to(self.unit)
                else:
                    #Temperature[uK]
                    signal = (cu.c**3*(1+self.halos_in_survey[line]['Ztrue'])**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo']/Vcell).to(self.unit)

                zmid = (self.line_nu0[line]/self.nuObs_mean).decompose().value-1
                sigma_par = (cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
                sigma_perp = (self.cosmo.comoving_radial_distance(zmid)*u.Mpc*(self.beam_width/(1*u.rad))).to(self.Mpch).value
                

                if self.pmeshpaint:
                    #Alternative painting using pmesh directly instead of nbk
                    pm = pmesh.pm.ParticleMesh(Nmesh, BoxSize=Lbox, dtype='float32', resampler='cic')
                    #Make realfield object
                    realfield = pm.create(type='real')
                    layout = pm.decompose(lategrid)
                    #Exchange positions between different MPI ranks 
                    p = layout.exchange(lategrid)
                    #Assign weights following the layout of particles
                    m = layout.exchange(signal.value)

                    pm.paint(p, out=realfield, mass=m, resampler='cic')

                    realfield = realfield.r2c()
                    #Fourier-transformed fields. 
                    realfield = realfield.apply(aniso_filter, kind='wavenumber')
                    #Now assumes maps is in Fourier-space
                    maps+=realfield
                else:
                #Build Nbodykit catalog object
                    nbodycat = np.empty(len(cartesian_halopos), dtype=[('Position', ('f8', 3)), ('Signal', 'f8')])
                    nbodycat['Position'] = lategrid 
                    nbodycat['Signal'] = signal.value 
                    cat = ArrayCatalog(nbodycat, Nmesh=Nmesh, BoxSize=Lbox)
                    #Convert to a mesh, weighting by signal
                    mesh = cat.to_mesh(Nmesh=Nmesh, BoxSize=Lbox, value='Signal',
                                       resampler='tsc',compensated=True,interlaced=True)
                #Apply the filtering to smooth mesh

                    mesh = mesh.apply(aniso_filter, mode='complex', kind='wavenumber')
                    #paint the map and resample to [Nchannel,Npix^0.5,Npix^0.5] (and rescale by change in volume)
                    maps += mesh.paint(mode='real')
        #Add the noise contribution 
        #noise_map = maps*0. + np.random.normal(0.,self.sigmaN.value,maps.shape)
        #zmid = (self.line_nu0[self.target_line]/self.nuObs_mean).decompose().value-1
        #sigma_par = (cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
        #sigma_perp = (self.cosmo.comoving_radial_distance(zmid)*u.Mpc*(self.beam_width/(1*u.rad))).to(self.Mpch).value
        #noise_map = noise_map.apply(aniso_filter, mode='complex', kind='wavenumber')
        #maps += noise_map.paint(mode='real')
                    maps += np.random.normal(0.,self.sigmaN.value,maps.shape)
        return maps
        
################################
## Compute summary statistics ##
################################

    @cached_survey_property
    def Pk_2d(self):
       '''
       Computes the 2d power spectrum P(k,mu) of the map
       '''
       return FFTPower(self.obs_map, '2d', Nmu=self.Nmu, poles=[0,2,4], los=[1,0,0], 
                       dk=self.dk.to(self.Mpch**-1).value,kmin=self.kmin.to(self.Mpch**-1).value) 
       
    @cached_survey_property
    def k_Pk_poles(self):
        '''
        Fourier wavenumbers for the multipoles of the power spectrum
        '''
        return self.Pk_2d.poles['k']
        
    @cached_survey_property
    def Pk_0(self):
        '''
        Monopole of the power spectrum
        '''
        return self.Pk_2d.poles['power_0'].real
        
    @cached_survey_property
    def Pk_2(self):
        '''
        Quadrupole of the power spectrum
        '''
        return self.Pk_2d.poles['power_2'].real
        
    @cached_survey_property
    def Pk_4(self):
        '''
        Hexadecapole of the power spectrum
        '''
        return self.Pk_2d.poles['power_4'].real
        
    @cached_survey_property
    def Ti(self):
        '''
        Center of the VID histogram bins
        '''
        if self.linear_VID_bin:
            Te = np.linespace(self.Tmin_VID.value,self.Tmax_VID.value,self.Nbin_hist+1)*self.Tmin_VID.unit
        else:
            Te = np.linespace(np.log10(self.Tmin_VID.value),np.log10(self.Tmax_VID.value),self.Nbin_hist+1)*self.Tmin_VID.unit
        Ti = (Te[:-1]+Te[1:])/2.
        return Ti
        
    @cached_survey_property
    def Bi_VID(self):
        '''
        Computes the histogram of temperatures in each voxel in hte observed map.
        Equivalent to the VID
        '''
        return 
    
        

                
#########################
## Auxiliary functions ##
#########################
    
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
                    
