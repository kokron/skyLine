'''
Base module to make a LIM survey from painted lightcone
'''

import numpy as np
import dask.array as da
import astropy.units as u
import astropy.constants as cu
from astropy.io import fits
import copy
import pmesh
import healpy as hp
from source.lightcone import Lightcone
from source.utilities import cached_survey_property,get_default_params,check_params
from warnings import warn

try:
    import pysm3
    NoPySM = False
except:
    NoPySM = True



class Survey(Lightcone):
    '''
    An object controlling all relevant quantities needed to create the
    painted LIM lightcone. It reads a lightcone catalog of halos with SFR
    quantities and paint it with as many lines as desired.

    It will be called by the class measure, to compute summary statistics

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

    -RAObs_width:           Total RA width observed. Assumed centered at 0 (Default = 2 deg)

    -DECObs_width:          Total DEC width observed. Assumed centered at 0 (Default = 2 deg)

    -dnu:                   Width of a single frequency channel (Default = 15.6 MHz)

    -beam_FWHM:             Beam full width at half maximum (Default = 4.1 arcmin)

    -tobs:                  Observing time on a single field (Default = 6000 hr)

    -target_line:           Target line of the survey (Default: CO)

    -angular_supersample:   Factor of supersample with respect to the survey angular resolution
                            when making the grid. Important to have good power spectrum multipole
                            calculations (e.g., enough mu values as function of kmax). (Default: 5)
                            
    -spectral_supersample:  Factor of supersample with respect to the survey spectral resolution
                            when making the grid. Important to have good power spectrum multipole
                            calculations (e.g., enough mu values as function of kmax). (Default: 1)

    -do_angular_smooth:     Boolean: apply smoothing filter to implement angular resolution
                            limitations. (Default: True)

    -do_spectral_smooth:    Boolean: apply smoothing filter to implement spectral resolution
                            limitations. (Default: False)
                            
    -kind_spectral_smooth:  Whether the spectral smoothing is done using a top-hat filter or a 
                            Gaussian filter ('tophat' or 'gaussian', respectively. Default: 'tophat')
                            
    -cube_mode:             Mode to create the data rectangular cube for 3d maps (irrelevant if 
                            do_angular == True). Options are:
                                - 'outer_cube': The lightcone is inscribed in the cube
                                - 'inner_cube': The cube is inscribed in the lightcone
                                - 'mid_redshift': coordinates transverse to line of sight are obtained
                                                  using a single redshift for all emitters (for each line)
                                - 'flat_sky': Applies flat sky approximation on top of 'mid_redshift'
                            (Default: 'inner_cube')
                            
    -do_z_buffering:        Boolean: add higher z emitters to fill corners at high-z end. Only relevant if 
                            cube_mode = 'inner_cube' or 'mid_redshift'. (Default: True)

    -do_downsample          Boolean: Downsample the map such as supersample=1.
                            (Default: True; make if False for nice plots)

    -do_remove_mean         Boolean: Remove the mean of the map or not
                            (Defult: True)

    -do_angular             Create an angular survey (healpy map)
                            (Default: False)

    -nside                  NSIDE used by healpy to create angular maps. (Default: 2048)

    -number_count           Boolean: Create a map with number density of haloes within the catalog.
                            It allows to use all galaxies or select between lrgs and elgs (defaul: False)

    -Mhalo_min              Minimum halo mass (in Msun/h) to be included in the survey (filter for halos_in_survey). Default:0

    -Mstar_min              Minimum stellar mass in a halo (in Msun) to be ncluded in the survey (filter for halos_in_survey). Default:0
    
    -gal_type               Whether to select only LRGs or ELGs, or all galaxies. Options: 'all', 'lrg', 'elg'.
    
    -resampler              Set the resampling window for the 3d maps (Irrelevant if do_angular=True). (Default: 'cic')

    '''
    def __init__(self,
                 do_intensity=False,
                 Tsys=40*u.K,
                 Nfeeds=19,
                 nuObs_min = 26.*u.GHz,
                 nuObs_max = 34.*u.GHz,
                 RAObs_width = 2.*u.deg,
                 DECObs_width = 2.*u.deg,
                 dnu=15.6*u.MHz,
                 beam_FWHM=4.1*u.arcmin,
                 tobs=6000*u.hr,
                 target_line = 'CO',
                 angular_supersample = 5,
                 spectral_supersample = 5,
                 do_angular_smooth = True,
                 do_spectral_smooth = False,
                 kind_spectral_smooth = 'tophat',
                 cube_mode = 'inner_cube',
                 do_z_buffering = True,
                 do_downsample = True,
                 do_remove_mean = True,
                 do_angular = False,
                 do_gal_foregrounds = False,
                 foreground_model=dict(precomputed_file=None, dgrade_nside=2**10, survey_center=[0*u.deg, 90*u.deg], sky={'synchrotron' : True, 'dust' : True, 'freefree' : True,'ame' : True}),
                 nside = 2048,
                 number_count=False,
                 Mhalo_min=0.,
                 Mstar_min=0.,
                 gal_type='all',
                 ngal=0.*u.Mpc**-3,
                 resampler='cic', 
                 **lightcone_kwargs):

        # Initiate Lightcone() parameters
        Lightcone.__init__(self,**lightcone_kwargs)

        self._update_lightcone_list = self._update_lightcone_list

        self._survey_params = locals()
        self._survey_params.pop('self')
        self._survey_params.pop('lightcone_kwargs')
        self._default_survey_params = get_default_params(Survey.__init__)
        check_params(self,self._survey_params,self._default_survey_params)

        # Set survey parameters
        for key in self._survey_params:
            setattr(self,key,self._survey_params[key])

        # Combine lightcone_params with survey_params
        self._input_params.update(self._survey_params)
        self._default_params.update(self._default_survey_params)
        
        #Limits for RA and DEC
        self.RAObs_min,self.RAObs_max = -self.RAObs_width/2.,self.RAObs_width/2.
        self.DECObs_min,self.DECObs_max = -self.DECObs_width/2.,self.DECObs_width/2.
        
        if self.RAObs_width.value == 360 and self.DECObs_width.value == 180:
            self.full_sky = True
        else:
            self.full_sky = False

        # Check that the observed footprint is contained in the lightcone
        if self.RAObs_min < self.RA_min or self.RAObs_max > self.RA_max or \
           self.DECObs_min < self.DEC_min or self.DECObs_max > self.DEC_max:
            warn('Please, your observed limits RA_Obs=[{},{}], DEC_Obs=[{},{}] must be within the lightcone limits RA=[{},{}], DEC=[{},{}].'.format(self.RAObs_min,self.RAObs_max,self.DECObs_min,self.DECObs_max,self.RA_min,self.RA_max,self.DEC_min,self.DEC_max))

        # Check that the bandwidth and lines used are included in the lightcone limits
        for line in self.lines.keys():
            if self.lines[line]:
                zlims = (self.line_nu0[line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
                if zlims[0] <= self.zmin or zlims [1] >= self.zmax:
                    warn('The line {} on the bandwidth [{:.2f},{:.2f}] corresponds to z range [{:.2f},{:.2f}], while the included redshifts in the lightcone are within [{:.2f},{:.2f}]. Please remove the line, increase the zmin,zmax range or reduce the bandwith.'.format(line,self.nuObs_max,self.nuObs_min,zlims[0],zlims[1],self.zmin,self.zmax))

        #Check healpy pixel size just in case:
        if self.do_angular:
            npix_fullsky = 4*np.pi/((self.beam_FWHM/self.angular_supersample)**2).to(u.sr).value
            min_nside = hp.pixelfunc.get_min_valid_nside(npix_fullsky)
            if (min_nside > self.nside):
                warn("The minimum NSIDE to account for beam_FWHM*angular_supersample is {}, but NSIDE={} was input.".format(min_nside,self.nside))

        self.cube_mode_options = ['outer_cube','inner_cube','mid_redshift','flat_sky']
        if self.cube_mode not in self.cube_mode_options:
            raise ValueError('The cube_mode choice must be one of {}'.format(self.cube_mode_options))
            
        if self.gal_type not in ['all','lrg','elg']:
            raise ValueError('gal_type input must be one of {}'.format(['all','lrg','elg']))
            
        if self.gal_type != 'all':
            if self.ngal.value == 0:
                raise ValueError('Please input a number density for ngal')
            if self.do_angular:
                try:
                    self.ngal = self.ngal.to(u.sr**-1)
                except:
                    raise ValueError('Please use angular number density for ngal if do_angular = True')
            else:
                try:
                    self.ngal = self.ngal.to(self.Mpch**-3)
                except:
                    raise ValueError('Please use 3D number density for ngal if do_angular = False')

        if NoPySM and self.do_gal_foregrounds==True:
            raise ValueError('PySM must be installed to model galactic foregrounds')

        #Set units for observable depending on convention
        if self.do_intensity:
            self.unit = u.Jy/u.sr
        else:
            self.unit = u.uK

        #Set global variables for smoothing kernel
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
        return ((self.line_nu0[self.target_line]/self.nuObs_mean).decompose()).value-1

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

        Assumes contiguous / simple survey geometry...
        '''
        phimax = self.RAObs_max.to(u.radian).value
        phimin = self.RAObs_min.to(u.radian).value
        thetamax = np.pi/2 - self.DECObs_max.to(u.radian).value
        thetamin = np.pi/2 - self.DECObs_min.to(u.radian).value
        
        omega = (phimax - phimin) * (np.cos(thetamax) - np.cos(thetamin))*u.sr
        return omega 

    @cached_survey_property
    def beam_width(self):
        '''
        Beam width defined as 1-sigma width of Gaussian beam profile
        '''
        return self.beam_FWHM*0.4247

    @cached_survey_property
    def Npixside(self):
        '''
        Number of pixels per side of the observed map. RA,DEC
        '''
        return int(np.round(((self.RAObs_width)/(self.beam_FWHM)).decompose())),\
               int(np.round(((self.DECObs_width)/(self.beam_FWHM)).decompose()))

    @cached_survey_property
    def Npix(self):
        '''
        Number of pixels in the observed map
        '''
        return self.Npixside[0]*self.Npixside[1]

    @cached_survey_property
    def Nchan(self):
        '''
        Number of frequency channels in the observed map
        '''
        return int(np.round((self.delta_nuObs/(self.dnu)).decompose()))
        
    @cached_survey_property
    def Vsurvey(self):
        '''
        Returns the comoving volume of the survey, computed from RA, DEC and nuObs limits
        for the target line
        '''
        #Omega_field * D_A (z)^2 * (1+z) * Delta_nu/nu_obs * c/H is the volume of the survey
            #D_A here is comoving angular diameter distance = comoving_radial_distance in flat space
        Area = self.Omega_field/u.sr*(self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc)**2
        Depth = self.delta_nuObs/(0.5*(self.nuObs_max+self.nuObs_min))*(1+self.zmid)*(cu.c.to('km/s')/self.cosmo.hubble_parameter(self.zmid)/(u.km/u.Mpc/u.s))
        return (Area*Depth).to(u.Mpc**3)

    @cached_survey_property
    def Lbox(self):
        '''
        Sides of the field observed (approximated to be a rectangular cube),
        for the assumed redshift (the one corresponding to the target line)
        '''
        #box angular limits (centered)
        ralim = np.deg2rad(np.array([self.RAObs_min.value,self.RAObs_max.value]))
        declim = np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value]))

        #transform Frequency band into redshift range for the target line
        zlims = (self.line_nu0[self.target_line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
        rlim = ((self.cosmo.comoving_radial_distance(zlims)*u.Mpc).to(self.Mpch)).value
        
        #projection to the unit sphere
        xlim = np.cos(declim) * np.cos(ralim)
        ylim = np.cos(declim) * np.sin(ralim)
        zlim = np.sin(declim)
        
        poscorner = np.vstack([xlim,ylim,zlim]).T
        corners = rlim[:,None]*poscorner[1] #All positive
        
        #Get the side of the box
        if self.cube_mode == 'inner_cube':
            raside = 2*rlim[0]*np.tan(0.5*(ralim[1]-ralim[0]))
            decside = 2*rlim[0]*np.tan(0.5*(declim[1]-declim[0]))
            zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
            
            self.raside_lim = rlim[0]*np.tan(ralim) #min, max 
            self.decside_lim = rlim[0]*np.tan(declim) #min, max
            self.rside_obs_lim = np.array([rlim[0],rlim[0]+zside]) #min, max
            
            warn("% of survey volume lost due to inner cube = {}".format(1-zside*raside*decside*self.Mpch**3/self.Vsurvey))
            
            if (corners[1,0] < rlim[0]+zside) and (self.do_z_buffering == False):
                warn("The corners of the last perpendicular slices of the box are going to be empty. Consider using 'inner_cube'=True or 'do_z_buffering'=True")
                
        elif self.cube_mode == 'outer_cube':
            raside = 2*rlim[1]*np.sin(0.5*(ralim[1]-ralim[0]))
            decside = 2*rlim[1]*np.sin(0.5*(declim[1]-declim[0]))
            zside = rlim[1]-corners[0,0]
            
            self.raside_lim = rlim[1]*np.sin(ralim) #min, max 
            self.decside_lim = rlim[1]*np.sin(declim) #min, max
            self.rside_obs_lim = np.array([rlim[1]-zside,rlim[1]]) #min, max
        
        elif self.cube_mode == 'flat_sky':
            rmid = ((self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc).to(self.Mpch)).value
            raside = 2*rmid*np.sin(0.5*(ralim[1]-ralim[0]))
            decside = 2*rmid*np.sin(0.5*(declim[1]-declim[0]))
            zside = rlim[1]-rlim[0]
            
            self.raside_lim = rmid*np.tan(ralim) #min, max 
            self.decside_lim = rmid*np.tan(declim) #min, max
            self.rside_obs_lim = np.array([rlim[0],rlim[1]]) #min, max
            
        elif self.cube_mode == 'mid_redshift':
            rmid = ((self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc).to(self.Mpch)).value
            raside = 2*rmid*np.sin(0.5*(ralim[1]-ralim[0]))
            decside = 2*rmid*np.sin(0.5*(declim[1]-declim[0]))
            #to avoid cut at high redshift end
            zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
            
            self.raside_lim = rmid*np.sin(ralim) #min, max 
            self.decside_lim = rmid*np.sin(declim) #min, max
            self.rside_obs_lim = np.array([rlim[0],rlim[0]+zside]) #min, max
            
            warn("% of survey volume lost due to cutting the spherical cap at high redshift end = {}".format(1-zside*raside*decside*self.Mpch**3/self.Vsurvey))
            
            if (corners[1,0] < rlim[0]+zside) and (self.do_z_buffering == False):
                warn("The corners of the last perpendicular slices of the box are going to be empty. Consider using 'inner_cube'=True or 'do_z_buffering'=True")
            
        Lbox = np.array([zside,raside,decside], dtype=np.float32)
        
        if np.any(Lbox) == 0:
            raise ValueError("The imposed cuts leave you with no volume, please review the observed RA, DEC and nu")

        return (Lbox*self.Mpch).to(self.Mpch)

    #########################
    ## Create the mock map ##
    #########################
    
    @cached_survey_property
    def halos_in_survey_all(self):
        '''
        Filters all the halo catalog and only takes those that have observed
        frequencies within the experimental frequency bandwitdh and lie in the
        observed RA - DEC ranges
        '''
        #empty catalog
        halos_survey = {}

        #halos within footprint
        if self.full_sky:
            inds_sky = np.ones(len(self.halo_catalog_all['RA']),dtype=bool)
        else:
            if self.do_angular:
                #Enhance the survey selection a bit to prevent healpy masking from giving limited objects at edges
                #Computes the mid-point of the boundaries and then expands them by 1%
                #May fail at low nside or weird survey masks
                inds_RA = (self.halo_catalog_all['RA'] > 0.995*self.RAObs_min.value)&(self.halo_catalog_all['RA'] < 1.005*self.RAObs_max.value)
                inds_DEC = (self.halo_catalog_all['DEC'] > 0.995*self.DECObs_min.value)&(self.halo_catalog_all['DEC'] < 1.005*self.DECObs_min.value)
            else:
                #make sure Lbox is run
                Lbox = self.Lbox
                inds_RA = (self.halo_catalog_all['RA'] > self.RAObs_min.value)&(self.halo_catalog_all['RA'] < self.RAObs_max.value)
                inds_DEC = (self.halo_catalog_all['DEC'] > self.DECObs_min.value)&(self.halo_catalog_all['DEC'] < self.DECObs_max.value)
            inds_sky = inds_RA&inds_DEC
            
        inds_mass = np.ones(len(inds_sky),dtype=bool)

        if self.Mhalo_min != 0.:
            inds_mass = inds_mass&(self.halo_catalog_all['M_HALO']>=self.Mhalo_min)
        if self.Mstar_min != 0.:
            inds_mass = inds_mass&(self.halo_catalog_all['SM_HALO']>=self.Mstar_min)
                            
        #Get a lower nu_Obs_min to buffer high redshifts and fill corners if required
        if (self.do_angular == False) and (self.do_z_buffering) and \
           (self.cube_mode == 'inner_cube' or self.cube_mode == 'mid_redshift'):
            cornerside = (self.raside_lim[1]**2+self.decside_lim[1]**2)**0.5
            ang = np.arctan(cornerside/self.rside_obs_lim[1])
            rbuffer = cornerside/np.sin(ang)
            zbuffer = self.cosmo.redshift_at_comoving_radial_distance((rbuffer*self.Mpch).value)
            nu_min = self.line_nu0[self.target_line]/(zbuffer+1)

            #print('The target line requires z_max = {:.3f} instead of the nominal {:.3f}'.format(zbuffer,(self.line_nu0[self.target_line]/self.nuObs_min).value-1))
            if zbuffer > self.zmax:
                warn('Filling the corners requires a buffering z_max = {:.3f}, but input z_max = {:.3f}. Corners will not be completely filled'.format(zbuffer,self.zmax))
        else:
            nu_min = self.nuObs_min

        #Loop over lines to see what halos are within nuObs
        for line in self.lines.keys():
            if self.lines[line]:
                halos_survey[line] = dict(RA= np.array([]),DEC=np.array([]),Zobs=np.array([]),Ztrue=np.array([]),Lhalo=np.array([])*u.Lsun,Mhalo=np.array([])*self.Msunh)
                    
                inds = (self.nuObs_line_halo_all[line] >= nu_min)&(self.nuObs_line_halo_all[line] <= self.nuObs_max)&inds_sky&inds_mass
                
                if self.gal_type != 'all' and self.number_count:
                    #separate between ELGs and LRGs
                    inds_gal = np.where((np.log10(self.halo_catalog_all['SM_HALO'])>8)&(self.halo_catalog_all['SFR_HALO']>0))
                    sSFR = self.halo_catalog_all['SFR_HALO'][inds_gal]/self.halo_catalog_all['SM_HALO'][inds_gal]
                    hist,bins = np.histogram(np.log10(sSFR),bins=101)
                    hist[hist==0] = 1e-100
                    hist = np.log10(hist)
                    inds_hist = [np.argmax(hist[:50]),np.argmax(hist[50:])+50]
                    indlim = np.argmin(hist[inds_hist[0]:inds_hist[1]])+1+inds_hist[0]
                    sSFR = self.halo_catalog_all['SFR_HALO']/self.halo_catalog_all['SM_HALO']
                    if self.gal_type == 'elg':
                        inds = inds&(sSFR > 10**bins[indlim])
                    else:
                        inds = inds&(sSFR < 10**bins[indlim])
                    #Get the N brightest (e.g., higher Mstar) up to matching number density
                    if self.do_angular:
                        Ngal_tot = self.ngal*self.Omega_field
                    else:
                        Ngal_tot = (self.ngal*((self.Lbox.value).prod()*(self.Mpch**3))).decompose()
                    Ngal_max = np.sum(inds)
                    if Ngal_tot > Ngal_max:
                        if self.do_angular:
                            ngal_max = np.sum(Ngal_max)/self.Omega_field
                        else:
                            ngal_max = np.sum(Ngal_max)/((self.Lbox.value).prod()*(self.Mpch**3))
                        warn("Maximum n_gal with the total number of {:}s is {:.5f}, input was {:5f}, reduce it or work with all {:}".format(self.gal_type,ngal_max,self.ngal,self.gal_type))
                    else:
                        argsort = np.argsort(self.halo_catalog_all['SM_HALO'])[::-1]
                        indlim = np.where(np.cumsum(inds[argsort])>Ngal_tot)[0][0]
                        inds[argsort[indlim:]] = False
                
                halos_survey[line]['RA'] = np.append(halos_survey[line]['RA'],self.halo_catalog_all['RA'][inds])
                halos_survey[line]['DEC'] = np.append(halos_survey[line]['DEC'],self.halo_catalog_all['DEC'][inds])
                halos_survey[line]['Zobs'] = np.append(halos_survey[line]['Zobs'],(self.line_nu0[self.target_line]/self.nuObs_line_halo_all[line][inds]).decompose()-1)
                # doing DZ correction
                halos_survey[line]['Ztrue'] = np.append(halos_survey[line]['Ztrue'],self.halo_catalog_all['Z'][inds]+self.halo_catalog_all['DZ'][inds])
                #halos_survey[line]['Ztrue'] = np.append(halos_survey[line]['Ztrue'],self.halo_catalog_all['Z'][inds])
                halos_survey[line]['Lhalo'] = np.append(halos_survey[line]['Lhalo'],self.L_line_halo_all[line][inds])
                halos_survey[line]['Mhalo'] = np.append(halos_survey[line]['Mhalo'],self.halo_catalog_all['M_HALO'][inds]*self.Msunh)

        return halos_survey

    def halos_in_survey_slice(self,line,nfiles,ifile):
        '''
        Filters the halo catalog and only takes those that have observed
        frequencies within the experimental frequency bandwitdh and lie in the
        observed RA - DEC ranges
        
        for a single slice, not cached
        '''
        #empty catalog
        halos_survey = {}
        
        #halos within footprint
        if self.full_sky:
            inds_sky = np.ones(len(self.halo_catalog['RA']),dtype=bool)
        else:
            if self.do_angular:
                #Enhance the survey selection a bit to prevent healpy masking from giving limited objects at edges
                #Computes the mid-point of the boundaries and then expands them by 1%
                #May fail at low nside or weird survey masks
                inds_RA = (self.halo_catalog['RA'] > 0.995*self.RAObs_min.value)&(self.halo_catalog['RA'] < 1.005*self.RAObs_max.value)
                inds_DEC = (self.halo_catalog['DEC'] > 0.995*self.DECObs_min.value)&(self.halo_catalog['DEC'] < 1.005*self.DECObs_min.value)
            else:
                #make sure Lbox is run
                Lbox = self.Lbox
                inds_RA = (self.halo_catalog['RA'] > self.RAObs_min.value)&(self.halo_catalog['RA'] < self.RAObs_max.value)
                inds_DEC = (self.halo_catalog['DEC'] > self.DECObs_min.value)&(self.halo_catalog['DEC'] < self.DECObs_max.value)
            inds_sky = inds_RA&inds_DEC
            
            
        inds_mass = np.ones(len(inds_sky),dtype=bool)

        if self.Mhalo_min != 0.:
            inds_mass = inds_mass&(self.halo_catalog['M_HALO']>=self.Mhalo_min)
        if self.Mstar_min != 0.:
            inds_mass = inds_mass&(self.halo_catalog['SM_HALO']>=self.Mstar_min)
            
        #Get a lower nu_Obs_min to buffer high redshifts and fill corners if required (for the last zbin)
        if ifile == nfiles-1:
            if (self.do_angular == False) and (self.do_z_buffering) and \
               (self.cube_mode == 'inner_cube' or self.cube_mode == 'mid_redshift'):
                cornerside = (self.raside_lim[1]**2+self.decside_lim[1]**2)**0.5
                ang = np.arctan(cornerside/self.rside_obs_lim[1])
                rbuffer = cornerside/np.sin(ang)
                zbuffer = self.cosmo.redshift_at_comoving_radial_distance((rbuffer*self.Mpch).value)
                nu_min = self.line_nu0[self.target_line]/(zbuffer+1)

                #print('The target line requires z_max = {:.3f} instead of the nominal {:.3f}'.format(zbuffer,(self.line_nu0[self.target_line]/self.nuObs_min).value-1))
                if zbuffer > self.zmax:
                    warn('Filling the corners requires a buffering z_max = {:.3f}, but input z_max = {:.3f}. Corners will not be completely filled'.format(zbuffer,self.zmax))
            else:
                nu_min = self.nuObs_min
        else:
            nu_min = self.nuObs_min

        #There's only halos from one line stored
        halos_survey[line] = dict(RA=np.array([]),DEC=np.array([]),Zobs=np.array([]),Ztrue=np.array([]),Lhalo=np.array([])*u.Lsun,Mhalo=np.array([])*self.Msunh)
        #get observed freqs and 
        self.nuObs_line_halo_slice(line)
        self.L_line_halo_slice(line)
        inds = (self.nuObs_line_halo[line] >= nu_min)&(self.nuObs_line_halo[line] <= self.nuObs_max)&inds_sky&inds_mass
        
        if self.gal_type != 'all':
            #separate between ELGs and LRGs
            inds_gal = np.where((np.log10(self.halo_catalog['SM_HALO'])>8)&(self.halo_catalog['SFR_HALO']>0))
            sSFR = self.halo_catalog['SFR_HALO'][inds_gal]/self.halo_catalog['SM_HALO'][inds_gal]
            hist,bins = np.histogram(np.log10(sSFR),bins=101)
            hist[hist==0] = 1e-100
            hist = np.log10(hist)
            inds_hist = [np.argmax(hist[:50]),np.argmax(hist[50:])+50]
            indlim = np.argmin(hist[inds_hist[0]:inds_hist[1]])+1+inds_hist[0]
            sSFR = self.halo_catalog['SFR_HALO']/self.halo_catalog['SM_HALO']
            if self.gal_type == 'elg':
                inds = inds&(sSFR > 10**bins[indlim])
            else:
                inds = inds&(sSFR < 10**bins[indlim])
            #Get the N brightest (e.g., higher Mstar) up to matching number density
            #**assuming that the number density of each file is equivalent...**
            if self.do_angular:
                Ngal_tot = self.ngal*self.Omega_field/nfiles
            else:
                Ngal_tot = (self.ngal*((self.Lbox.value).prod()*(self.Mpch**3))).decompose()/nfiles
            Ngal_max = np.sum(inds)
            if Ngal_tot > Ngal_max:
                if self.do_angular:
                    ngal_max = np.sum(Ngal_max)/self.Omega_field
                else:
                    ngal_max = np.sum(Ngal_max)/((self.Lbox.value).prod()*(self.Mpch**3))
                warn("Maximum n_gal with the total number of {:}s in this slice is {:.5f}, input was {:5f}, reduce it or work with all {:}".format(self.gal_type,ngal_max,self.ngal,self.gal_type))
            else:
                argsort = np.argsort(self.halo_catalog['SM_HALO'])[::-1]
                indlim = np.where(np.cumsum(inds[argsort])>Ngal_tot)[0][0]
                inds[argsort[indlim:]] = False
                
        halos_survey[line]['RA'] = np.append(halos_survey[line]['RA'],self.halo_catalog['RA'][inds])
        halos_survey[line]['DEC'] = np.append(halos_survey[line]['DEC'],self.halo_catalog['DEC'][inds])
        halos_survey[line]['Zobs'] = np.append(halos_survey[line]['Zobs'],(self.line_nu0[self.target_line]/self.nuObs_line_halo[line][inds]).decompose()-1)
        #doing DZ correction
        halos_survey[line]['Ztrue'] = np.append(halos_survey[line]['Ztrue'],self.halo_catalog['Z'][inds]+self.halo_catalog['DZ'][inds])
        #halos_survey[line]['Ztrue'] = np.append(halos_survey[line]['Ztrue'],self.halo_catalog['Z'][inds])
        halos_survey[line]['Lhalo'] = np.append(halos_survey[line]['Lhalo'],self.L_line_halo[line][inds])
        halos_survey[line]['Mhalo'] = np.append(halos_survey[line]['Mhalo'],self.halo_catalog['M_HALO'][inds]*self.Msunh)

        self.halos_in_survey = halos_survey


    @cached_survey_property
    def obs_2d_map(self):
        '''
        Generates the mock intensity map observed in spherical shells. It does not include noise.
        '''
        if not self.do_angular:
            warn('Mask edges will be funky in this case, might see some vignetting')
        npix = hp.nside2npix(self.nside)
        hp_map = np.zeros(npix)

        # First, compute the intensity/temperature of each halo in the catalog we will include
        for line in self.lines.keys():
            if self.lines[line]:
                if not self.cache_catalog:
                    #get zmin zmax for the line and the files
                    zmin_line = ((self.line_nu0[line]/self.nuObs_max).decompose()).value-1
                    zmax_line = ((self.line_nu0[line]/self.nuObs_min).decompose()).value-1
                    #add some buffer to be sure
                    fnames = self.halo_slices(zmin_line-0.03,zmax_line+0.03)
                    nfiles = len(fnames)
                                    
                    for ifile in range(nfiles):
                        #Get the halos and which of those fall in the survey
                        self.halo_catalog_slice(fnames[ifile])
                        self.halos_in_survey_slice(line,nfiles,ifile)
                        #add the contribution from these halos
                        hp_map = self.paint_2d(self.halos_in_survey[line],line,hp_map)
                else:
                    hp_map = self.paint_2d(self.halos_in_survey_all[line],line,hp_map)
                    
        # add galactic foregrounds
        if self.do_gal_foregrounds:
            hp_map+=self.create_2d_foreground_map()

        #smooth for angular resolution
        if self.do_angular_smooth:
            theta_beam = self.beam_FWHM.to(u.rad)
            hp_map = hp.smoothing(hp_map, theta_beam.value)

        #get the proper nside for the observed map
        if self.do_downsample:
            npix_fullsky = 4*np.pi/(self.beam_FWHM**2).to(u.sr).value
            nside_min = hp.pixelfunc.get_min_valid_nside(npix_fullsky)
            if nside_min < self.nside:
                hp_map = hp.ud_grade(hp_map,nside_min)

        #Define the mask from the rectangular footprint if not full sky!
        if not self.full_sky:
            #padding to avoid errors
            pad_ra,pad_dec = 0,0
            if self.RAObs_width == 360:
                pad_ra = 1e-5
            if self.DECObs_width == 180:
                pad_dec = 1e-5
            phicorner_list = np.linspace(self.RAObs_min.value+pad_ra,self.RAObs_max.value-pad_ra,10)
            thetacorner = np.pi/2-np.deg2rad(np.array([self.DECObs_min.value+pad_dec,self.DECObs_max.value-pad_dec,self.DECObs_max.value-pad_dec,self.DECObs_min.value+pad_dec]))
            pix_within = np.array([])
            for iphiedge in range(len(phicorner_list)-1):
                phicorner = np.deg2rad(np.array([phicorner_list[iphiedge],phicorner_list[iphiedge],phicorner_list[iphiedge+1],phicorner_list[iphiedge+1]]))
                vecs = hp.dir2vec(thetacorner,phi=phicorner).T
                try:
                    pix_within = np.append(pix_within,hp.query_polygon(nside=self.nside,vertices=vecs,inclusive=False))
                except:
                    pix_within = np.append(pix_within, [])
            self.pix_within = pix_within
            mask = np.ones(hp.nside2npix(self.nside),np.bool)
            mask[pix_within.astype(int)] = 0
            hp_map = hp.ma(hp_map)
            hp_map.mask = mask
        
        #remove the monopole
        if self.do_remove_mean:
            hp_map = hp.pixelfunc.remove_monopole(hp_map,copy=False)

        return hp_map
        
    def paint_2d(self,halos,line,hp_map):
        '''
        Adds the contribution from a slice to the 2d healpy map
        '''
        #Get true cell volume
        #Get positions using the observed redshift
        ra,dec,redshift = da.broadcast_arrays(halos['RA'], halos['DEC'],
                                              halos['Zobs'])

        Zhalo = halos['Ztrue']
        Hubble = self.cosmo.hubble_parameter(Zhalo)*(u.km/u.Mpc/u.s)

        #Figure out what channel the halos will be in to figure out the voxel volume, for the signal.
        #This is what will be added to the healpy map.
        nu_bins = self.nuObs_min.to('GHz').value + np.arange(self.Nchan)*self.dnu.to('GHz').value
        zmid_channel = self.line_nu0[line].to('GHz').value/(nu_bins + 0.5*self.dnu.to('GHz').value) - 1

        #Channel of each halo, can now compute voxel volumes where each of them are seamlessly
        bin_idxs = np.digitize(self.line_nu0[line].to('GHz').value/(1+Zhalo), nu_bins)-1
        zmids = zmid_channel[bin_idxs]

        #Vcell = Omega_pix * D_A (z)^2 * (1+z) * dnu/nu_obs * c/H is the volume of the voxel for a given channel
                            #D_A here is comoving angular diameter distance = comoving_radial_distance in flat space
        Vcell_true = hp.nside2pixarea(self.nside)*(self.cosmo.comoving_radial_distance(zmids)*u.Mpc )**2 * (self.dnu.value/nu_bins[bin_idxs]) * (1+zmids) * (cu.c.to('km/s')/self.cosmo.hubble_parameter(zmids)/(u.km/u.Mpc/u.s))

        if not self.number_count:
            if self.do_intensity:
                #intensity[Jy/sr]
                signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*halos['Lhalo']/Vcell_true).to(self.unit)
            else:
                #Temperature[uK]
                signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*halos['Lhalo']/Vcell_true).to(self.unit)
        else:
            #number counts [empty unit]
            signal = np.ones(len(Zhalo))*(1*self.unit/self.unit)
            
        #Paste the signals to the map
        theta, phi = rd2tp(halos['RA'], halos['DEC'])
        pixel_idxs = hp.ang2pix(self.nside, theta, phi)

        np.add.at(hp_map, pixel_idxs, signal.value)
        
        return hp_map
   

    @cached_survey_property
    def obs_3d_map(self):
        '''
        Generates the mock intensity map observed in Fourier space,
        obtained from Cartesian coordinates. It does not include noise.
        '''
        #Define the mesh divisions and the box size
        zmid = (self.line_nu0[self.target_line]/self.nuObs_mean).decompose().value-1
        sigma_par_target = (cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
        
        Lbox = self.Lbox.value
        
        Nmesh = np.array([self.spectral_supersample*np.ceil(Lbox[0]/sigma_par_target),
                  self.angular_supersample*self.Npixside[0],
                  self.angular_supersample*self.Npixside[1]], dtype=int)
        
        ralim = np.deg2rad(np.array([self.RAObs_min.value,self.RAObs_max.value])) 
        declim = np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value]))
        raside_lim = self.raside_lim
        decside_lim = self.decside_lim
        rside_obs_lim = self.rside_obs_lim

        #Setting the box with the origin at 0 plus additional padding to get voxel coordinates at their center
        mins_obs = np.array([rside_obs_lim[0],raside_lim[0],decside_lim[0]])#+0.49999*Lbox/Nmesh #I think not needed in the end?

        global sigma_par
        global sigma_perp
        maps = np.zeros([Nmesh[0],Nmesh[1],Nmesh[2]//2 + 1], dtype='complex64')


        # First, compute the intensity/temperature of each halo in the catalog we will include
        for line in self.lines.keys():
            if self.lines[line]:
                #Create the mesh
                pm = pmesh.pm.ParticleMesh(Nmesh, BoxSize=Lbox, dtype='float32', resampler=self.resampler)
                #Make realfield object
                field = pm.create(type='real')
                field[:] = 0.
                
                #Get true cell volume
                zlims = (self.line_nu0[line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
                rlim = ((self.cosmo.comoving_radial_distance(zlims)*u.Mpc).to(self.Mpch)).value
                #Get the side of the box
                #projection to the unit sphere
                xlim = np.cos(declim) * np.cos(ralim)
                ylim = np.cos(declim) * np.sin(ralim)
                zlim = np.sin(declim)

                poscorner = np.vstack([xlim,ylim,zlim]).T
                corners = rlim[:,None]*poscorner[1] #All positive

                #Get the side of the box
                if self.cube_mode == 'inner_cube':
                    raside = 2*rlim[0]*np.tan(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rlim[0]*np.tan(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
                    rmid = 0

                elif self.cube_mode == 'outer_cube':
                    raside = 2*rlim[1]*np.sin(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rlim[1]*np.sin(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]-corners[0,0]
                    rmid = 0

                elif self.cube_mode == 'flat_sky':
                    zmid = ((self.line_nu0[line]/self.nuObs_mean).decompose()).value-1
                    rmid = ((self.cosmo.comoving_radial_distance(zmid)*u.Mpc).to(self.Mpch)).value
                    raside = 2*rmid*np.sin(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rmid*np.sin(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]-rlim[0]
                    
                elif self.cube_mode == 'mid_redshift':
                    zmid = ((self.line_nu0[line]/self.nuObs_mean).decompose()).value-1
                    rmid = ((self.cosmo.comoving_radial_distance(zmid)*u.Mpc).to(self.Mpch)).value
                    raside = 2*rmid*np.sin(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rmid*np.sin(0.5*(declim[1]-declim[0]))
                    #to avoid cut at high redshift end
                    zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
                    
                Lbox_true = np.array([zside,raside,decside])
                Vcell_true = (Lbox_true/Nmesh).prod()*(self.Mpch**3).to(self.Mpch**3)
                
                if not self.cache_catalog:
                    #add some buffer to be sure
                    fnames = self.halo_slices(zlims[0]-0.03,zlims[1]+0.03)
                    nfiles = len(fnames)
                    for ifile in range(nfiles):
                        #Get the halos and which of those fall in the survey
                        self.halo_catalog_slice(fnames[ifile])
                        self.halos_in_survey_slice(line,nfiles,ifile)
                        #add the contribution from these halos
                        field += self.paint_3d(self.halos_in_survey[line],line,rmid,mins_obs,Vcell_true,pm)
                else:
                    field += self.paint_3d(self.halos_in_survey_all[line],line,rmid,mins_obs,Vcell_true,pm)
                
        #turn the field to complex
        field = field.r2c()
        #This smoothing comes from the resolution window function.
        if self.do_spectral_smooth or self.do_angular_smooth:
            #compute scales for the anisotropic filter (in Ztrue -> zmid)
            zmid = (self.line_nu0[self.target_line]/self.nuObs_mean).decompose().value-1
            sigma_par = self.do_spectral_smooth*(cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
            sigma_perp = self.do_angular_smooth*(self.cosmo.comoving_radial_distance(zmid)*u.Mpc*(self.beam_width/(1*u.rad))).to(self.Mpch).value
            if self.kind_spectral_smooth == 'tophat':
                field = field.apply(aniso_filter_tophat_los, kind='wavenumber')
            elif self.kind_spectral_smooth == 'gaussian':
                field = field.apply(aniso_filter_gaussian_los, kind='wavenumber')
        #Add this contribution to the total maps
        maps+=field

        # add galactic foregrounds
        if self.do_gal_foregrounds:
            maps+=self.create_3d_foreground_map(mins_obs, Nmesh, Lbox, rside_obs_lim, raside_lim, decside_lim)

        #get the proper shape for the observed map
        if (self.angular_supersample > 1 or self.spectral_supersample > 1) and self.do_downsample:
            pm_down = pmesh.pm.ParticleMesh(np.array([self.Nchan,self.Npixside[0],self.Npixside[1]], dtype=int),
                                                  BoxSize=Lbox, dtype='float32', resampler=self.resampler)
            maps = pm_down.downsample(maps.c2r(),keep_mean=True)
        else:
            maps = maps.c2r()
        
        #Remove mean
        if self.do_remove_mean:
            maps = maps-maps.cmean()

        return maps
        
    def paint_3d(self,halos,line,rmid,mins_obs,Vcell_true,pm):
        '''
        Adds the contribution from a slice to the 3d pmesh map
        '''
        #Get positions using the observed redshift
        #Convert the halo position in each volume to Cartesian coordinates (from Nbodykit)
        ra,dec,redshift = da.broadcast_arrays(halos['RA'], halos['DEC'],
                                              halos['Zobs'])
        
        #radial distances in Mpch/h
        r = redshift.map_blocks(lambda zz: (((self.cosmo.comoving_radial_distance(zz)*u.Mpc).to(self.Mpch)).value),
                                dtype=redshift.dtype)

        ra,dec  = da.deg2rad(ra),da.deg2rad(dec)
        if self.cube_mode == 'flat_sky':
            # cartesian coordinates in flat sky
            x = da.ones(ra.shape[0])
            y = ra/r*rmid 
            z = dec/r*rmid 
        elif self.cube_mode == 'mid_redshift':
            # cartesian coordinates in unit sphere but preparing for only one distance for ra and dec
            x = da.cos(dec) * da.cos(ra)
            y = da.sin(ra)/r*rmid # only ra?
            z = da.sin(dec)/r*rmid # only dec?
        else:
            # cartesian coordinates in unit sphere
            x = da.cos(dec) * da.cos(ra)
            y = da.cos(dec) * da.sin(ra)
            z = da.sin(dec)
            
        pos = da.vstack([x,y,z]).T                    
        cartesian_halopos = r[:,None] * pos
        lategrid = np.array(cartesian_halopos.compute())
        #Filter some halos out if outside of the cube mode
        if self.cube_mode == 'inner_cube' or self.cube_mode == 'mid_redshift':
            filtering = (lategrid[:,0] >= self.rside_obs_lim[0]) & (lategrid[:,0] <= self.rside_obs_lim[1]) & \
                        (lategrid[:,1] >= self.raside_lim[0]) & (lategrid[:,1] <= self.raside_lim[1]) & \
                        (lategrid[:,2] >= self.decside_lim[0]) & (lategrid[:,2] <= self.decside_lim[1])
            lategrid = lategrid[filtering]
            #Compute the signal in each voxel (with Ztrue and Vcell_true)
            Zhalo = halos['Ztrue'][filtering]
            Hubble = self.cosmo.hubble_parameter(Zhalo)*(u.km/u.Mpc/u.s)
            
            warn("% of emitters of {} line left out filtering = {}".format(line, 1-len(Zhalo)/len(filtering)))

            if not self.number_count:
                if self.do_intensity:
                    #intensity[Jy/sr]
                    signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*halos['Lhalo'][filtering]/Vcell_true).to(self.unit)
                else:
                    #Temperature[uK]
                    signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*halos['Lhalo'][filtering]/Vcell_true).to(self.unit)
        else:
            Zhalo = halos['Ztrue']
            Hubble = self.cosmo.hubble_parameter(Zhalo)*(u.km/u.Mpc/u.s)
            if not self.number_count:
                if self.do_intensity:
                    #intensity[Jy/sr]
                    signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*halos['Lhalo']/Vcell_true).to(self.unit)
                else:
                    #Temperature[uK]
                    signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*halos['Lhalo']/Vcell_true).to(self.unit)
        #Locate the grid such that bottom left corner of the box is [0,0,0] which is the nbodykit convention.
        for n in range(3):
            lategrid[:,n] -= mins_obs[n] 
            
        #Make realfield temp object
        tempfield = pm.create(type='real')
        tempfield[:] = 0.
                
        #Set the emitter in the grid and paint using pmesh directly instead of nbk
        layout = pm.decompose(lategrid)
        #Exchange positions between different MPI ranks
        p = layout.exchange(lategrid)
        #Assign weights following the layout of particles
        if self.number_count:
            nbar = len(p)/np.prod(pm.Nmesh)
            pm.paint(p, out=tempfield, mass=1/nbar, resampler=self.resampler)
        else:
            m = layout.exchange(signal.value)
            pm.paint(p, out=tempfield, mass=m, resampler=self.resampler)
                    
        return tempfield
    
    @cached_survey_property
    def noise_3d_map(self):
        '''
        3d map with the instrumental noise in the cosmic volume probed by target line
        '''
        #add the noise, distribution is gaussian with 0 mean
        if self.do_downsample:
            noise_map = self.rng.normal(0.,self.sigmaN.value,self.obs_3d_map.shape)
        else:
            supersample_sigmaN = self.sigmaN * (self.spectral_supersample)**0.5 * self.angular_supersample
            noise_map = self.rng.normal(0.,supersample_sigmaN.value,self.obs_3d_map.shape)

        return noise_map
    
    @cached_survey_property
    def noise_2d_map(self):
        '''
        2d angular map with the instrumental noise in the cosmic volume probed by target line
        '''
        #rescale the noise per pixel to the healpy pixel size
        hp_sigmaN = self.sigmaN * (self.pix_within.size/self.Npix)**0.5
        #add the noise, distribution is gaussian with 0 mean
        noise_map = np.zeros(len(self.obs_2d_map))
        noise_map[pix_within] = self.rng.normal(0.,hp_sigmaN.value,pix_within.size)

        return noise_map
                
    def vec2pix_func(self, x, y, z):
        '''
        Alias for hp.vec2pix(nside,x,y,z) function to use in the cartesian projection
        '''
        return hp.vec2pix(self.nside, x, y, z)

    def create_3d_foreground_map(self, mins, Nmesh, Lbox, rside_obs_lim, raside_lim, decside_lim):
        '''
        Creates a 3D map of galactic continuum foregrounds using pySM
        '''
        if self.foreground_model['dgrade_nside']!=self.nside:
            dgrade_nside=self.foreground_model['dgrade_nside']
        else:
            dgrade_nside=self.nside
                          
        #TO DO: ADD case for z_buffering option and check foreground for other projections
        if self.do_angular == False and self.cube_mode != 'flat_sky':
            warn('Careful! The implementation of the foregrounds has only been thoroughly tested for flat sky projections yet.')
        
        if (self.do_angular == False) and (self.do_z_buffering) and (self.cube_mode == 'inner_cube' or self.cube_mode == 'mid_redshift'):
            cornerside = (self.raside_lim[1]**2+self.decside_lim[1]**2)**0.5
            ang = np.arctan(cornerside/self.rside_obs_lim[1])
            rbuffer = cornerside/np.sin(ang)
            zbuffer = self.cosmo.redshift_at_comoving_radial_distance((rbuffer*self.Mpch).value)
            nu_min = self.line_nu0[self.target_line]/(zbuffer+1)
        else:
            nu_min = self.nuObs_min
        obs_freqs_edge=np.linspace(nu_min, self.nuObs_max, self.spectral_supersample*self.Nchan+1)
        obs_freqs=(obs_freqs_edge[1:]+obs_freqs_edge[:-1])/2 #frequencies observed in survey

        if self.foreground_model['precomputed_file']!=None:
            ra_insurvey=[]; dec_insurvey=[]; z_insurvey=[]; foreground_signal=[]
            for i in range(len(obs_freqs)):
                dgrade_galmap_rotated=hp.fitsfunc.read_map(self.foreground_model['precomputed_file'][i]) #read pre-computed healpy maps
                if self.foreground_model['dgrade_nside']!=self.nside:
                    galmap_rotated=hp.pixelfunc.ud_grade(dgrade_galmap_rotated, self.nside)
                else:
                    galmap_rotated=dgrade_galmap_rotated
                    
                if self.do_angular_smooth:
                    theta_beam = self.beam_FWHM.to(u.rad)
                    galmap_rotated = hp.smoothing(galmap_rotated, theta_beam.value)

                ramin=(self.RAObs_min.value)
                ramax=(self.RAObs_max.value)
                decmin=(self.DECObs_min.value)
                decmax=(self.DECObs_max.value)

                cart_proj=hp.projector.CartesianProj(xsize=self.Npixside[0]*self.angular_supersample, ysize=self.Npixside[1]*self.angular_supersample, lonra =  [ramin,ramax], latra=[decmin,decmax])  
                galmap_cart=cart_proj.projmap(galmap_rotated, self.vec2pix_func)
                foreground_signal.append((galmap_cart.flatten())*self.unit)
               
                Xedge=np.linspace(ramin,ramax, (self.Npixside[0]*self.angular_supersample)+1)
                Yedge=np.linspace(decmin,decmax, (self.Npixside[1]*self.angular_supersample)+1)
                X=(Xedge[1:]+Xedge[:-1])/2
                Y=(Yedge[1:]+Yedge[:-1])/2
                Xpix,Ypix=np.meshgrid(X,Y)
                Xpix=Xpix.flatten()
                Ypix=Ypix.flatten()

                ra_insurvey.append(Xpix)
                dec_insurvey.append(Ypix)
                z_insurvey.append((self.line_nu0[self.target_line]/obs_freqs[i] -1)*np.ones((len(Xpix))))
        else:
            #build foreground component dictionary
            components=[key for key,value in self.foreground_model['sky'].items() if value == True]
            sky_config = []
            for cmp in components:
                if cmp=='synchrotron':
                    sky_config.append("s1")
                elif cmp=='dust':
                    sky_config.append("d1")
                elif cmp=='freefree':
                    sky_config.append("f1")
                elif cmp=='ame':
                    sky_config.append("a1")
                else:
                    warn('Unknown galactic foreground component: {}'.format(cmp))

            sky = pysm3.Sky(nside=dgrade_nside, preset_strings=sky_config)#create sky object using the specified model
            ra_insurvey=[]; dec_insurvey=[]; z_insurvey=[]; foreground_signal=[]
            for i in range(len(obs_freqs)):
                dgrade_galmap=sky.get_emission(obs_freqs[i])[0]#produce healpy maps, 0 index corresponds to intensity
                rot_center = hp.Rotator(rot=[self.foreground_model['survey_center'][0].to_value(u.deg), self.foreground_model['survey_center'][1].to_value(u.deg)], inv=True) #rotation to place the center of the survey at the origin              
                if self.do_angular_smooth:
                    theta_beam = self.beam_FWHM
                else:
                    theta_beam = 0*u.arcmin
                dgrade_galmap_rotated = pysm3.apply_smoothing_and_coord_transform(dgrade_galmap, rot=rot_center, fwhm=theta_beam)
                if self.foreground_model['dgrade_nside']!=self.nside:
                    galmap_rotated=hp.pixelfunc.ud_grade(dgrade_galmap_rotated, self.nside)
                else:
                    galmap_rotated=dgrade_galmap_rotated
                     
                ramin=(self.RAObs_min.value)
                ramax=(self.RAObs_max.value)
                decmin=(self.DECObs_min.value)
                decmax=(self.DECObs_max.value)

                cart_proj=hp.projector.CartesianProj(xsize=self.Npixside[0]*self.angular_supersample, ysize=self.Npixside[1]*self.angular_supersample, lonra =  [ramin,ramax], latra=[decmin,decmax])  
                galmap_cart=cart_proj.projmap(galmap_rotated, self.vec2pix_func)
                foreground_signal.append((galmap_cart.flatten()))
               
                Xedge=np.linspace(ramin,ramax, (self.Npixside[0]*self.angular_supersample)+1)
                Yedge=np.linspace(decmin,decmax, (self.Npixside[1]*self.angular_supersample)+1)
                X=(Xedge[1:]+Xedge[:-1])/2
                Y=(Yedge[1:]+Yedge[:-1])/2
                Xpix,Ypix=np.meshgrid(X,Y)
                Xpix=Xpix.flatten()
                Ypix=Ypix.flatten()

                ra_insurvey.append(Xpix)
                dec_insurvey.append(Ypix)
                z_insurvey.append((self.line_nu0[self.target_line]/obs_freqs[i] -1)*np.ones((len(Xpix))))

        ra,dec,redshift = da.broadcast_arrays(np.asarray(ra_insurvey).flatten(), np.asarray(dec_insurvey).flatten(), np.asarray(z_insurvey).flatten())
        #radial distances in Mpch/h
        r = redshift.map_blocks(lambda zz: (((self.cosmo.comoving_radial_distance(zz)*u.Mpc).to(self.Mpch)).value),
                                dtype=redshift.dtype)

        ra,dec  = da.deg2rad(ra),da.deg2rad(dec)
        if self.cube_mode == 'flat_sky':
            rmid = ((self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc).to(self.Mpch)).value
            # cartesian coordinates in flat sky
            x = da.ones(ra.shape[0])
            y = ra/r*rmid 
            z = dec/r*rmid 
        elif self.cube_mode == 'mid_redshift':
            rmid = ((self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc).to(self.Mpch)).value
            # cartesian coordinates in unit sphere but preparing for only one distance for ra and dec
            x = da.cos(dec) * da.cos(ra)
            y = da.sin(ra)/r*rmid # only ra?
            z = da.sin(dec)/r*rmid # only dec?
        else:
            # cartesian coordinates in unit sphere
            x = da.cos(dec) * da.cos(ra)
            y = da.cos(dec) * da.sin(ra)
            z = da.sin(dec)
        
        pos = da.vstack([x,y,z]).T
        cartesian_pixelpos = r[:,None] * pos
        foreground_grid = np.array(cartesian_pixelpos.compute())
        
        if self.cube_mode == 'inner_cube' or self.cube_mode == 'mid_redshift':
            filtering = (foreground_grid[:,0] >= rside_obs_lim[0]) & (foreground_grid[:,0] <= rside_obs_lim[1]) & \
                        (foreground_grid[:,1] >= raside_lim[0]) & (foreground_grid[:,1] <= raside_lim[1]) & \
                        (foreground_grid[:,2] >= decside_lim[0]) & (foreground_grid[:,2] <= decside_lim[1])
            foreground_grid = foreground_grid[filtering]
            foreground_signal=np.asarray(foreground_signal).flatten()[filtering]

        for n in range(3):
            foreground_grid[:,n] -= mins[n]
                    
        #Set the emitter in the grid and paint using pmesh directly instead of nbk
        pm = pmesh.pm.ParticleMesh(Nmesh, BoxSize=Lbox, dtype='float32', resampler=self.resampler)
        #Make realfield object
        field = pm.create(type='real')
        layout = pm.decompose(foreground_grid,smoothing=0.5*pm.resampler.support)
        #Exchange positions between different MPI ranks
        p = layout.exchange(foreground_grid)
        #Assign weights following the layout of particles
        m = layout.exchange(np.asarray(foreground_signal).flatten())
        pm.paint(p, out=field, mass=m, resampler=self.resampler)
        #Fourier transform fields and apply the filter
        field = field.r2c()
        return field
        
        
    def create_2d_foreground_map(self):
        '''
        Creates a 2D map of galactic continuum foregrounds using pySM
        '''
        if self.foreground_model['dgrade_nside']!=self.nside:
            dgrade_nside=self.foreground_model['dgrade_nside']
        else:
            dgrade_nside=self.nside
            
        hp_fg_map = np.zeros(hp.nside2npix(self.nside))

        if self.foreground_model['precomputed_file']!=None:
            for i in range(len(self.foreground_model['precomputed_file'])):
                dgrade_galmap_rotated=hp.fitsfunc.read_map(self.foreground_model['precomputed_file'][i]) #read pre-computed healpy maps
                if self.foreground_model['dgrade_nside']!=self.nside:
                    galmap_rotated=hp.pixelfunc.ud_grade(dgrade_galmap_rotated, self.nside)
                else:
                    galmap_rotated=dgrade_galmap_rotated
                    
                hp_fg_map += galmap_rotated
        else:
            #build foreground component dictionary
            components=[key for key,value in self.foreground_model['sky'].items() if value == True]
            sky_config = []
            for cmp in components:
                if cmp=='synchrotron':
                    sky_config.append("s1")
                elif cmp=='dust':
                    sky_config.append("d1")
                elif cmp=='freefree':
                    sky_config.append("f1")
                elif cmp=='ame':
                    sky_config.append("a1")
                else:
                    warn('Unknown galactic foreground component: {}'.format(cmp))
                    
            obs_freqs_edge=np.linspace(self.nuObs_min, self.nuObs_max, self.Nchan+1)
            obs_freqs=(obs_freqs_edge[1:]+obs_freqs_edge[:-1])/2 #frequencies observed in survey

            sky = pysm3.Sky(nside=dgrade_nside, preset_strings=sky_config)#create sky object using the specified model
            #produce healpy maps, 0 index corresponds to intensity
            dgrade_galmap=sky.get_emission(obs_freqs)[0]#produce healpy maps, 0 index corresponds to intensity
            rot_center = hp.Rotator(rot=[self.foreground_model['survey_center'][0].to_value(u.deg), self.foreground_model['survey_center'][1].to_value(u.deg)], inv=True) #rotation to place the center of the survey at the origin              
            dgrade_galmap_rotated = pysm3.apply_smoothing_and_coord_transform(dgrade_galmap, rot=rot_center, fwhm=0*u.arcmin)
            if self.foreground_model['dgrade_nside']!=self.nside:
                galmap_rotated=hp.pixelfunc.ud_grade(dgrade_galmap_rotated, self.nside)
            else:
                galmap_rotated=dgrade_galmap_rotated
            if self.do_intensity and sky.output_unit != self.unit:
                galmap_rotated *= pysm3.bandpass_unit_conversion(obs_freqs, input_unit = sky.output_unit,output_unit=self.unit)
            elif self.do_intensity == False and sky.output_unit != pysm3.units.uK_RJ:
                galmap_rotated *= pysm3.bandpass_unit_conversion(obs_freqs, input_unit = sky.output_unit,output_unit=self.unit)
                
            hp_fg_map += galmap_rotated
            
            return hp_fg_map
                
                
    def save_map(self,name,other_map=None):
        '''
        Saves a map (either pmesh or healpy depending on do_angular).
        If other_map != None, the map saved would be self.obs_map
        '''
        if not other_map:
            map_to_save = self.obs_map
        else:
            map_to_save = other_map
        if self.do_angular:
            hp.fitsfunc.write_map(name,map_to_save)
        else:
            hdu = fits.PrimaryHDU(map_to_save)
            hdu.writeto(name)
        return


#########################
## Auxiliary functions ##
#########################

def aniso_filter_tophat_los(k, v):
    '''
    Filter for k_perp and k_par modes separately.
    Uses a top-hat filter (sinc in Fourier space) for the k_par modes
    Applies to an nbodykit mesh object as a regular filter.

    Uses globally defined variables:
        sigma_perp - 'angular' smoothing in the flat sky approximation
        sigma_par - 'radial' smoothing from number of channels.

    Usage:
        mesh.apply(perp_filter, mode='complex', kind='wavenumber')

    NOTES:
    k[0] *= modifies the next iteration in the loop.
    Coordinates are fixed except for the k[1] which are
    the coordinate that sets what slab is being altered

    '''
    rper = sigma_perp
    rpar = sigma_par
    newk = copy.deepcopy(k)
    
    kk2_perp = newk[1]**2 + newk[2]**2
    
    #np.sinc(x) = sin(x*pi)/(x*pi); np.sinc(0) = 1
    w = np.exp(-0.5*kk2_perp * rper**2)*np.sinc(newk[0]*rpar/2/np.pi)

    #w[newk[0] == 0] = 1.0
    return w*v


def aniso_filter_gaussian_los(k, v):
    '''
    Filter for k_perp and k_par modes separately.
    Uses a gaussian filter for the k_par modes
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



def rd2tp(ra,dec):
    """ convert ra/dec to theta,phi"""

    phi = ra*np.pi/180

    theta = np.pi/180 * (90. - dec)
    return theta, phi

def tp2ra(theta, phi):
    """ convert theta,phi to ra,dec"""

    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)
    return ra, dec


def observed_mask_2d(self):
    pix = np.arange(hp.nside2npix(self.nside), dtype=int)
    theta, phi = hp.pix2ang(self.nside, pix)
    ra, dec = tp2ra(theta, phi)
    ra[ra>180]=ra[ra>180]-360

    RAmask=(ra>=self.RAObs_min.value)&(ra<self.RAObs_max.value)
    DECmask=(dec>=self.DECObs_min.value)&(dec<self.DECObs_max.value)

    mask=RAmask&DECmask
    return ra, dec, mask

def build_inv_mask(self, dgrade_nside):
    _, _, square_mask=observed_mask_2d(self)
    pix_mask = np.arange(hp.nside2npix(self.nside), dtype=int)[square_mask]
    square_partial_map = hp.UNSEEN*np.ones((hp.nside2npix(self.nside)))
    square_partial_map[pix_mask] = np.ones((len(pix_mask)))

    dgrade_square_partial_map=hp.pixelfunc.ud_grade(square_partial_map, dgrade_nside)

    rot = hp.Rotator(rot=[0, -90], inv=True)
    rotated_partial_map = rot.rotate_map_pixel(dgrade_square_partial_map)
    return hp.pixelfunc.mask_good(rotated_partial_map)

def build_partial_map(self, pixel_indices, pixel_values, dgrade_nside):
    partial_map = np.nan * np.empty((self.Nchan, hp.nside2npix(dgrade_nside)))
    partial_map[:, pixel_indices] = pixel_values
    return partial_map
