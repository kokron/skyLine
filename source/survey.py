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

    -RAObs_min,RAObs_max:   minimum and maximum RA observed (Default = -65-60 deg)

    -DECObs_min,DECObs_max: minimum and maximum DEC observed (Default = -1.25-1.25 deg)

    -dnu:                   Width of a single frequency channel (Default = 15.6 MHz)

    -beam_FWHM:             Beam full width at half maximum (Default = 4.1 arcmin)

    -tobs:                  Observing time on a single field (Default = 6000 hr)

    -target_line:           Target line of the survey (Default: CO)

    -angular_supersample:   Factor of supersample with respect to the survey angular resolution
                            when making the grid. (Default: 5)
                            
    -spectral_supersample:  Factor of supersample with respect to the survey spectral resolution
                            when making the grid. (Default: 1)

    -do_angular_smooth:     Boolean: apply smoothing filter to implement angular resolution
                            limitations. (Default: True)

    -do_spectral_smooth:    Boolean: apply smoothing filter to implement spectral resolution
                            limitations. (Default: False)
                            
    -cube_mode:             Mode to create the data rectangular cube for 3d maps (irrelevant if 
                            do_angular == True). Options are:
                                - 'outer_cube': The lightcone is inscribed in the cube
                                - 'inner_cube': The cube is inscribed in the lightcone
                                - 'mid_redshift': coordinates transverse to line of sight are obtained
                                                  using a single redshift for all emitters (for each line)
                                - 'flat_sky': Applies flat sky approximation on top of 'mid_redshift'
                            (Default: 'inner_cube')
    
    do_flat_sky:           Boolean: Use flat-sky approximation when building 3d map
                            (Default: False) Irrelevant if do_angular == True

    -do_inner_cut           Get a box for which there are no empty spaces, but discards some haloes.
                            (Default: True). Irrelevant if do_flat_sky == True or do_angular == True

    -do_downsample          Boolean: Downsample the map such as supersample=1.
                            (Default: True; make if False for nice plots)

    -do_remove_mean         Boolean: Remove the mean of the map or not
                            (Defult: True)

    -do_angular             Create an angular survey (healpy map)
                            (Default: False)

    -average_angular_proj   Average total integrated intensity per the number of channels for
                            angular projections (Default: True)

    -nside                  NSIDE used by healpy to create angular maps. (Default: 2048)

    -mass                   Boolean: Create a map with number density of ALL the haloes within the catalog (defaul: False)

    -Mhalo_min              Minimum halo mass (in Msun/h) to be included in the survey (filter for halos_in_survey). Default:0

    -Mstar_min              Minimum stellar mass in a halo (in Msun) to be ncluded in the survey (filter for halos_in_survey). Default:0

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
                 angular_supersample = 5,
                 spectral_supersample = 5,
                 do_angular_smooth = True,
                 do_spectral_smooth = False,
                 cube_mode = 'inner_cube',
                 do_downsample = True,
                 do_remove_mean = True,
                 do_angular = False,
                 do_gal_foregrounds = False,
                 foreground_model=dict(precomputed_file=None, dgrade_nside=2**10, survey_center=[0*u.deg, 90*u.deg], sky={'synchrotron' : True, 'dust' : True, 'freefree' : True,'ame' : True}),
                 average_angular_proj = True,
                 nside = 2048,
                 mass=False,
                 Mhalo_min=0.,
                 Mstar_min=0.,
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
            # ~ #Avoid inner cut if do_angular:
            # ~ if self.do_angular and self.do_inner_cut and not self.do_flat_sky:
                # ~ warn('If you want to work with angular maps, you do not need the inner cut, hence please use do_inner_cut = False')
        
        self.cube_mode_options = ['outer_cube','inner_cube','mid_redshift','flat_sky']
        if self.cube_mode not in self.cube_mode_options:
            raise ValueError('The cube_mode choice must be one of {}'.format(self.cube_mode_options))

        if NoPySM and self.do_gal_foregrounds==True:
            raise ValueError('PySM must be installed to model galactic foregrounds')

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
    def Npixside(self):
        '''
        Number of pixels per side of the observed map. RA,DEC
        '''
        return int(np.round(((self.RAObs_max-self.RAObs_min)/(self.beam_FWHM)).decompose())),\
               int(np.round(((self.DECObs_max-self.DECObs_min)/(self.beam_FWHM)).decompose()))

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
    def sigmaN(self):
        '''
        Instrumental voxel/pixel (depending on do_angular) noise standard deviation
        '''
        tpix = self.tobs/self.Npix
        if self.do_intensity:
            #intensity[Jy/sr]
            sig2 = self.Tsys**2/(self.Nfeeds*tpix)
        else:
            #Temperature[uK]
            sig2 = self.Tsys**2/(self.Nfeeds*self.dnu*tpix)

        if self.do_angular and self.average_angular_proj:
            sig2 /= self.Nchan

        return (sig2**0.5).to(self.unit)
        
    @cached_survey_property
    def Vsurvey(self):
        '''
        Returns the comoving volume of the survey, computed from RA, DEC and nuObs limits
        for the target line
        '''
        #Omega_field * D_A (z)^2 * (1+z) * Delta_nu/nu_obs * c/H is the volume of the survey
            #D_A here is comoving angular diameter distance = comoving_radial_distance in flat space
        Vol = self.Omega_field*(self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc)**2 * (self.delta_nuObs/self.line_nu0[self.target_line]) * (cu.c.to('km/s')/self.cosmo.hubble_parameter(self.zmid)*(u.km/u.Mpc/u.s))
        return Vol.to(u.Mpc**3)

    @cached_survey_property
    def Lbox(self):
        '''
        Sides of the field observed (approximated to be a rectangular cube),
        for the assumed redshift (the one corresponding to the target line)
        '''
        #box center
        ramid = np.deg2rad(0.5*(self.RAObs_max + self.RAObs_min).value)
        decmid = np.deg2rad(0.5*(self.DECObs_max + self.DECObs_min).value)
        
        #box angular limits (centered)
        ralim = np.deg2rad(np.array([self.RAObs_min.value,self.RAObs_max.value])) - ramid
        declim = np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value])) - decmid

        #transform Frequency band into redshift range for the target line
        zlims = (self.line_nu0[self.target_line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
        rlim = ((self.cosmo.comoving_radial_distance(zlims)*u.Mpc).to(self.Mpch)).value
        
        #Get the side of the box
        if self.cube_mode == 'inner_cube':
            raside = 2*rlim[0]*np.tan(0.5*(ralim[1]-ralim[0]))
            decside = 2*rlim[0]*np.tan(0.5*(declim[1]-declim[0]))
            zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
            
            self.rside_obs_lim = np.array([rlim[0],rlim[0]+zside]) #min, max
            
            warn("% of survey volume lost due to inner cube = {}".format(1-zside*raside*decside*self.Mpch**3/self.Vsurvey))
            
            if (raside > np.sqrt(2)*2*rlim[1]*np.sin(0.5*(ralim[1]-ralim[0]))) or \
               (decside > np.sqrt(2)*2*rlim[1]*np.sin(0.5*(declim[1]-declim[0]))):
                warn("The corners of the last perpendicular slices of the box are going to be empty. Consider using flat sky")
                
        elif self.cube_mode == 'outer_cube':
            raside = 2*rlim[1]*np.tan(0.5*(ralim[1]-ralim[0]))
            decside = 2*rlim[1]*np.tan(0.5*(declim[1]-declim[0]))
            zside = rlim[1]-rlim[0]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))
            
            self.rside_obs_lim = np.array([rlim[1]-zside,rlim[1]]) #min, max
        
        elif self.cube_mode == 'flat_sky':
            rmid = ((self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc).to(self.Mpch)).value
            raside = 2*rmid*np.tan(0.5*(ralim[1]-ralim[0]))
            decside = 2*rmid*np.tan(0.5*(declim[1]-declim[0]))
            zside = rlim[1]-rlim[0]
            
            self.rside_obs_lim = np.array([rlim[0],rlim[1]]) #min, max
            
        elif self.cube_mode == 'mid_redshift':
            rmid = ((self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc).to(self.Mpch)).value
            raside = 2*rmid*np.tan(0.5*(ralim[1]-ralim[0]))
            decside = 2*rmid*np.tan(0.5*(declim[1]-declim[0]))
            #to avoid cut at high redshift end
            zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
            
            self.rside_obs_lim = np.array([rlim[0],rlim[0]+zside]) #min, max
            
            warn("% of survey volume lost due to cutting the spherical cap at high redshift end = {}".format(1-zside*raside*decside*self.Mpch**3/self.Vsurvey))
            
            if (raside > np.sqrt(2)*2*rlim[1]*np.sin(0.5*(ralim[1]-ralim[0])) or \
               (decside > np.sqrt(2)*2*rlim[1]*np.sin(0.5*(declim[1]-declim[0])):
                warn("The corners of the last perpendicular slices of the box are going to be empty. Consider using flat sky")

        self.raside_lim = rlim[0]*np.tan(ralim) #min, max self.decside_lim = rlim[0]*np.tan(declim-decmid) #min, max
        self.decside_lim = rlim[0]*np.tan(declim) #min, max
            
        Lbox = np.array([zside,raside,decside])

        return (Lbox*self.Mpch).to(self.Mpch)

    # ~ @cached_survey_property
    # ~ def Vvox(self):
        # ~ '''
        # ~ Voxel volume (approximated as a rectangular cube) for the assumed redshift
        # ~ (the one corresponding to the target line).

        # ~ Voxel dimensions given by the resolution of the experiment. 
        # ~ '''
        # ~ Nmesh = np.array([self.Nchan,self.Npixside[0], self.Npixside[1]], dtype=int)
        # ~ #return (self.Lbox.value/Nmesh).prod()*self.Lbox.unit**3
        # ~ return (self.Lbox.value/Nmesh).prod()*self.Lbox.unit**3


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
        if self.do_angular:
            #Enhance the survey selection a bit to prevent healpy masking from giving limited objects at edges
            #Computes the mid-point of the boundaries and then expands them by 1%
            #May fail at low nside or weird survey masks
            delta_ra = 1.01*0.5*(self.RAObs_max.value - self.RAObs_min.value)
            mid_ra = 0.5*(self.RAObs_max.value + self.RAObs_min.value)

            delta_dec = 1.01*0.5*(self.DECObs_max.value - self.DECObs_min.value)
            mid_dec = 0.5*(self.DECObs_max.value + self.DECObs_min.value)

            inds_RA = (self.halo_catalog['RA'] > mid_ra - delta_ra)&(self.halo_catalog['RA'] < delta_ra  +mid_ra)
            inds_DEC = (self.halo_catalog['DEC'] > mid_dec - delta_dec)&(self.halo_catalog['DEC'] < mid_dec + delta_dec)
        else:
            inds_RA = (self.halo_catalog['RA'] > self.RAObs_min.value)&(self.halo_catalog['RA'] < self.RAObs_max.value)
            inds_DEC = (self.halo_catalog['DEC'] > self.DECObs_min.value)&(self.halo_catalog['DEC'] < self.DECObs_max.value)
        inds_sky = inds_RA&inds_DEC
        inds_mass = np.ones(len(inds_sky),dtype=bool)

        if self.Mhalo_min != 0.:
            inds_mass = inds_mass&(self.halo_catalog['M_HALO']>=self.Mhalo_min)
        if self.Mstar_min != 0.:
            inds_mass = inds_mass&(self.halo_catalog['SM_HALO']>=self.Mstar_min)

        #Loop over lines to see what halos are within nuObs
        for line in self.lines.keys():
            if self.lines[line]:
                halos_survey[line] = dict(RA= np.array([]),DEC=np.array([]),Zobs=np.array([]),Ztrue=np.array([]),Lhalo=np.array([])*u.Lsun)
                #inds = (self.nuObs_line_halo[line] >= self.nuObs_min)&(self.nuObs_line_halo[line] <= self.nuObs_max)&inds_sky
                inds = (self.nuObs_line_halo[line] >= self.nuObs_min)&(self.nuObs_line_halo[line] <= self.nuObs_max)&inds_sky&inds_mass
                halos_survey[line]['RA'] = np.append(halos_survey[line]['RA'],self.halo_catalog['RA'][inds])
                halos_survey[line]['DEC'] = np.append(halos_survey[line]['DEC'],self.halo_catalog['DEC'][inds])
                halos_survey[line]['Zobs'] = np.append(halos_survey[line]['Zobs'],(self.line_nu0[self.target_line]/self.nuObs_line_halo[line][inds]).decompose()-1)
                #Not doing DZ correction
                #halos_survey[line]['Ztrue'] = np.append(halos_survey[line]['Ztrue'],self.halo_catalog['Z'][inds]+self.halo_catalog['DZ'][inds])
                halos_survey[line]['Ztrue'] = np.append(halos_survey[line]['Ztrue'],self.halo_catalog['Z'][inds])
                halos_survey[line]['Lhalo'] = np.append(halos_survey[line]['Lhalo'],self.L_line_halo[line][inds])

        return halos_survey


    @cached_survey_property
    def obs_2d_map(self):
        '''
        Generates the mock intensity map observed in spherical shells. It does not include noise.
        '''
        #Define the mesh divisions and the box size

        if not self.do_angular:
            warn('Mask edges will be funky in this case, might see some vignetting')
        npix = hp.nside2npix(self.nside)

        #This is too much memory
        # maps = np.zeros((self.Nchan, npix))

        hp_map = np.zeros(npix)

        # First, compute the intensity/temperature of each halo in the catalog we will include
        for line in self.lines.keys():
            if self.lines[line]:
                hp_map_line = np.zeros(npix)

                #Get true cell volume
                #Get positions using the observed redshift
                ra,dec,redshift = da.broadcast_arrays(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'],
                                                      self.halos_in_survey[line]['Zobs'])

                Zhalo = self.halos_in_survey[line]['Ztrue']
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
                Vcell_true = hp.nside2pixarea(self.nside)*(self.cosmo.comoving_radial_distance(zmids)*u.Mpc )**2 * (self.dnu/self.line_nu0[line]) * (cu.c.to('km/s')/self.cosmo.hubble_parameter(zmids)*(u.km/u.Mpc/u.s))

                if not self.mass:
                    if self.do_intensity:
                        #intensity[Jy/sr]
                        signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                    else:
                        #Temperature[uK]
                        signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                else:
                    #number counts [empty unit]
                    signal = np.ones(len(Zhalo))*(1*self.unit/self.unit)

                #Paste the signals to the map
                theta, phi = rd2tp(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'])
                pixel_idxs = hp.ang2pix(self.nside, theta, phi)

                if self.average_angular_proj:
                    #averaging over the number of channels
                    np.add.at(hp_map_line, pixel_idxs, signal.value/self.Nchan)
                else:
                    np.add.at(hp_map_line, pixel_idxs, signal.value)
                #should smoothing be after masking?
                #could lead to bleeding of the zeros with the boundary
                if self.do_angular_smooth:
                    theta_beam = self.beam_FWHM.to(u.rad)
                    hp_map_line = hp.smoothing(hp_map_line, theta_beam.value)
                hp_map += hp_map_line

        #get the proper nside for the observed map
        if self.do_downsample:
            npix_fullsky = 4*np.pi/(self.beam_FWHM**2).to(u.sr).value
            nside_min = hp.pixelfunc.get_min_valid_nside(npix_fullsky)
            if nside_min < self.nside:
                hp_map = hp.ud_grade(hp_map,nside_min)

        #Define the mask from the rectangular footprint
        phicorner_list = np.append(np.arange(self.RAObs_min.value,self.RAObs_max.value,1),self.RAObs_max.value)
        thetacorner = np.pi/2-np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value,self.DECObs_max.value,self.DECObs_min.value]))
        pix_within = np.array([])
        for iphiedge in range(len(phicorner_list)-1):
            phicorner = np.deg2rad(np.array([phicorner_list[iphiedge],phicorner_list[iphiedge],phicorner_list[iphiedge+1],phicorner_list[iphiedge+1]]))
            vecs = hp.dir2vec(thetacorner,phi=phicorner).T
            pix_within = np.append(pix_within,hp.query_polygon(nside=self.nside,vertices=vecs,inclusive=False))
        self.pix_within = pix_within
        mask = np.ones(hp.nside2npix(self.nside),np.bool)
        mask[pix_within.astype(int)] = 0
        hp_map = hp.ma(hp_map)
        hp_map.mask = mask

        #add noise
        if self.Tsys.value > 0.:
            #rescale the noise per pixel to the healpy pixel size
            hp_sigmaN = self.sigmaN * (pix_within.size/self.Npix)**0.5
            if self.average_angular_proj:
                hp_sigmaN *= 1./(self.Nchan)**0.5
            hp_map[pix_within] += self.rng.normal(0.,hp_sigmaN.value,pix_within.size)

        #remove the monopole
        if self.do_remove_mean:
            hp_map = hp.pixelfunc.remove_monopole(hp_map,copy=False)

        return hp_map


    @cached_survey_property
    def obs_3d_map(self):
        '''
        Generates the mock intensity map observed in Fourier space,
        obtained from Cartesian coordinates. It does not include noise.
        '''

        if self.do_angular:
            warn('Mask edges might be problematic due to the expanded selection!')

        #Define the mesh divisions and the box size
        Nmesh = np.array([self.spectral_supersample*self.Nchan,
                  self.angular_supersample*self.Npixside[0],
                  self.angular_supersample*self.Npixside[1]], dtype=int)
        Lbox = self.Lbox.value

        ramid = 0.5*(self.RAObs_max + self.RAObs_min)
        decmid = 0.5*(self.DECObs_max + self.DECObs_min)
        
        ralim = np.deg2rad(np.array([self.RAObs_min.value,self.RAObs_max.value]) - ramid.value) 
        declim = np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value]) - decmid.value)
        raside_lim = self.raside_lim
        decside_lim = self.decside_lim
        rside_obs_lim = self.rside_obs_lim

        mins_obs = np.array([rside_obs_lim[0],raside_lim[0],decside_lim[0]])

        global sigma_par
        global sigma_perp
        maps = np.zeros([Nmesh[0],Nmesh[1],Nmesh[2]//2 + 1], dtype='complex64')


        # First, compute the intensity/temperature of each halo in the catalog we will include
        for line in self.lines.keys():
            if self.lines[line]:
                #Get true cell volume
                zlims = (self.line_nu0[line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
                rlim = ((self.cosmo.comoving_radial_distance(zlims)*u.Mpc).to(self.Mpch)).value
                #Get the side of the box
                if self.cube_mode == 'inner_cube':
                    raside = 2*rlim[0]*np.tan(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rlim[0]*np.tan(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
                        
                elif self.cube_mode == 'outer_cube':
                    raside = 2*rlim[1]*np.tan(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rlim[1]*np.tan(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]-rlim[0]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))
                    
                elif self.cube_mode == 'flat_sky':
                    zmid = (self.line_nu0[line]/self.nuObs_mean).decompose()-1
                    rmid = ((self.cosmo.comoving_radial_distance(zmid)*u.Mpc).to(self.Mpch)).value
                    raside = 2*rmid*np.tan(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rmid*np.tan(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]-rlim[0]
                    
                elif self.cube_mode == 'mid_redshift':
                    zmid = (self.line_nu0[line]/self.nuObs_mean).decompose()-1
                    rmid = ((self.cosmo.comoving_radial_distance(zmid)*u.Mpc).to(self.Mpch)).value
                    raside = 2*rmid*np.tan(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rmid*np.tan(0.5*(declim[1]-declim[0]))
                    #to avoid cut at high redshift end
                    zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]

                Lbox_true = np.array([zside,raside,decside])
                Vcell_true = (Lbox_true/Nmesh).prod()*(self.Mpch**3).to(self.Mpch**3)
                #Get positions using the observed redshift
                #Convert the halo position in each volume to Cartesian coordinates (from Nbodykit)
                ra,dec,redshift = da.broadcast_arrays(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'],
                                                      self.halos_in_survey[line]['Zobs'])
                
                #radial distances in Mpch/h
                r = redshift.map_blocks(lambda zz: (((self.cosmo.comoving_radial_distance(zz)*u.Mpc).to(self.Mpch)).value),
                                        dtype=redshift.dtype)
                                        
                #Shift the ra and dec of the halo such that they are centered in (0,0)
                ra -= ramid.value
                dec -= decmid.value

                ra,dec  = da.deg2rad(ra),da.deg2rad(dec)
                if self.cube_mode == 'flat_sky':
                    # cartesian coordinates in flat sky
                    x = da.ones(ra.shape[0])
                    y = da.sin(ra)/r*rmid # only ra?
                    z = da.sin(dec)/r*rmid # only dec?
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
                    filtering = (lategrid[:,0] >= rside_obs_lim[0]) & (lategrid[:,0] <= rside_obs_lim[1]) & \
                                (lategrid[:,1] >= raside_lim[0]) & (lategrid[:,1] <= raside_lim[1]) & \
                                (lategrid[:,2] >= decside_lim[0]) & (lategrid[:,2] <= decside_lim[1])
                    lategrid = lategrid[filtering]
                    #Compute the signal in each voxel (with Ztrue and Vcell_true)
                    Zhalo = self.halos_in_survey[line]['Ztrue'][filtering]
                    Hubble = self.cosmo.hubble_parameter(Zhalo)*(u.km/u.Mpc/u.s)
                    
                    warn("% of emitters of {} line left out filtering = {}".format(line, 1-len(Zhalo)/len(filtering))

                    if not self.mass:
                        if self.do_intensity:
                            #intensity[Jy/sr]
                            signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo'][filtering]/Vcell_true).to(self.unit)
                        else:
                            #Temperature[uK]
                            signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo'][filtering]/Vcell_true).to(self.unit)
                else:
                    Zhalo = self.halos_in_survey[line]['Ztrue']
                    Hubble = self.cosmo.hubble_parameter(Zhalo)*(u.km/u.Mpc/u.s)
                    if not self.mass:
                        if self.do_intensity:
                            #intensity[Jy/sr]
                            signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                        else:
                            #Temperature[uK]
                            signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                #Locate the grid such that bottom left corner of the box is [0,0,0] which is the nbodykit convention.
                for n in range(3):
                    lategrid[:,n] -= mins_obs[n]
                #Set the emitter in the grid and paint using pmesh directly instead of nbk
                pm = pmesh.pm.ParticleMesh(Nmesh, BoxSize=Lbox, dtype='float32', resampler='cic')
                #Make realfield object
                field = pm.create(type='real')
                layout = pm.decompose(lategrid)
                #Exchange positions between different MPI ranks
                p = layout.exchange(lategrid)
                #Assign weights following the layout of particles
                if self.mass:
                    pm.paint(p, out=field, mass=1, resampler='cic')
                else:
                    m = layout.exchange(signal.value)
                    pm.paint(p, out=field, mass=m, resampler='cic')
                #Fourier transform fields and apply the filter
                field = field.r2c()
                #This smoothing comes from the resolution window function.
                if self.do_spectral_smooth or self.do_angular_smooth:
                    #compute scales for the anisotropic filter (in Ztrue -> zmid)
                    zmid = (self.line_nu0[line]/self.nuObs_mean).decompose().value-1
                    sigma_par = self.do_spectral_smooth*(cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
                    sigma_perp = self.do_angular_smooth*(self.cosmo.comoving_radial_distance(zmid)*u.Mpc*(self.beam_width/(1*u.rad))).to(self.Mpch).value
                    field = field.apply(aniso_filter, kind='wavenumber')
                #Add this contribution to the total maps
                maps+=field

        # add galactic foregrounds
        if self.do_gal_foregrounds:
            field=self.create_foreground_map(mins, Nmesh, Lbox, rside_obs_lim, raside_lim, decside_lim)
            maps+=field

        #get the proper shape for the observed map
        if (self.angular_supersample > 1 or self.spectral_supersample > 1) and self.do_downsample:
            pm_down = pmesh.pm.ParticleMesh(np.array([self.Nchan,self.Npixside[0],self.Npixside[1]], dtype=int),
                                                  BoxSize=Lbox, dtype='float32', resampler='cic')
            maps = pm_down.downsample(maps.c2r(),keep_mean=True)
        else:
            maps = maps.c2r()

        #Add noise in the cosmic volume probed by target line to the 3d maps
        if self.Tsys.value > 0.:
            #add the noise, distribution is gaussian with 0 mean
            if self.do_downsample:
                maps += self.rng.normal(0.,self.sigmaN.value,maps.shape)
            else:
                supersample_sigmaN = self.sigmaN * (self.spectral_supersample)**0.5 * self.angular_supersample
                maps += self.rng.normal(0.,supersample_sigmaN.value,maps.shape)

        #Remove mean
        if self.do_remove_mean:
            maps = maps-maps.cmean()

        return maps

    def vec2pix_func(self, x, y, z):
        '''
        Alias for hp.vec2pix(nside,x,y,z) function to use in the cartesian projection
        '''
        return hp.vec2pix(self.nside, x, y, z)

    def create_foreground_map(self, mins, Nmesh, Lbox, rside_obs_lim, raside_lim, decside_lim):
        '''
        Creates a map of galactic continuum foregrounds using pySM
        '''
        if self.foreground_model['dgrade_nside']!=self.nside:
            dgrade_nside=self.foreground_model['dgrade_nside']
        else:
            dgrade_nside=self.nside
        obs_freqs_edge=np.linspace(self.nuObs_min, self.nuObs_max, self.spectral_supersample*self.Nchan+1)
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

                    #Compute the signal in each voxel (with Ztrue and Vcell_true)
                ramid = 0.5*(self.RAObs_max + self.RAObs_min)
                decmid = 0.5*(self.DECObs_max + self.DECObs_min)  
                ramin=self.RAObs_min.value-ramid.value-(self.RAObs_max-self.RAObs_min).value/self.Npixside[0]
                ramax=self.RAObs_max.value-ramid.value+(self.RAObs_max-self.RAObs_min).value/self.Npixside[0]
                decmin=self.DECObs_min.value-decmid.value-(self.DECObs_max-self.DECObs_min).value/self.Npixside[1]
                decmax=self.DECObs_max.value-decmid.value+(self.DECObs_max-self.DECObs_min).value/self.Npixside[1]

                cart_proj=hp.projector.CartesianProj(xsize=self.Npixside[0]*self.angular_supersample+2, ysize=self.Npixside[1]*self.angular_supersample+2, lonra =  [ramin,ramax], latra=[decmin,decmax])  
                galmap_cart=cart_proj.projmap(galmap_rotated, self.vec2pix_func)
                foreground_signal.append((galmap_cart.flatten())*u.uK)
               
                Xedge=np.linspace(ramin,ramax, self.Npixside[0]*self.angular_supersample+1+2)
                Yedge=np.linspace(decmin,decmax, self.Npixside[1]*self.angular_supersample+1+2)
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
                    
                cart_proj=hp.projector.CartesianProj(xsize=self.Npixside[0]*self.angular_supersample, ysize=self.Npixside[1]*self.angular_supersample, lonra =  [self.RAObs_min.value,self.RAObs_max.value], latra=[self.DECObs_min.value,self.DECObs_max.value])
                galmap_cart=cart_proj.projmap(galmap_rotated, self.vec2pix_func)
                foreground_signal.append((galmap_cart.flatten())*u.uK)
               
                Xedge=np.linspace(self.RAObs_min.value,self.RAObs_max.value, self.Npixside[0]*self.angular_supersample+1)
                Yedge=np.linspace(self.DECObs_min.value,self.DECObs_max.value, self.Npixside[1]*self.angular_supersample+1)
                X=(Xedge[1:]+Xedge[:-1])/2
                Y=(Yedge[1:]+Yedge[:-1])/2
                Xpix,Ypix=np.meshgrid(X,Y)
                Xpix=Xpix.flatten()
                Ypix=Ypix.flatten()

                ra_insurvey.append(Xpix)
                dec_insurvey.append(Ypix)
                z_insurvey.append((self.line_nu0[self.target_line]/obs_freqs[i] -1)*np.ones((len(Xpix))))

            #norm=hp.nside2pixarea(self.nside, degrees=True)*(u.deg**2).to(self.Omega_field.unit)/(self.Omega_field/self.Npix)
            #ra_fullsky, dec_fullsky, obs_mask= observed_mask_2d(self)
            #ra_insurvey=[]; dec_insurvey=[]; z_insurvey=[]; foreground_signal=[]
            #for i in range(len(obs_freqs)):
            #    dgrade_galmap=sky.get_emission(obs_freqs[i])[0]#produce healpy maps, 0 index corresponds to intensity   
            #    rot_center = hp.Rotator(rot=[self.foreground_model['survey_center'][0].to_value(u.deg), self.foreground_model['survey_center'][1].to_value(u.deg)], inv=True) #rotation to place the center of the survey at the origin
            #    dgrade_galmap_rotated = pysm3.apply_smoothing_and_coord_transform(dgrade_galmap, rot=rot_center)
            #    if self.foreground_model['dgrade_nside']!=self.nside:
            #        galmap_rotated=hp.pixelfunc.ud_grade(dgrade_galmap_rotated, self.nside)
            #    else:
            #        galmap_rotated=dgrade_galmap_rotated
            #    if self.do_angular_smooth:
            #        theta_beam = self.beam_FWHM.to(u.rad)
            #        galmap_rotated = hp.smoothing(galmap_rotated, theta_beam.value)

        ra,dec,redshift = da.broadcast_arrays(np.asarray(ra_insurvey).flatten(), np.asarray(dec_insurvey).flatten(), np.asarray(z_insurvey).flatten())
        #radial distances in Mpch/h
        r = redshift.map_blocks(lambda zz: (((self.cosmo.comoving_radial_distance(zz)*u.Mpc).to(self.Mpch)).value),
                                dtype=redshift.dtype)
                                
        #Shift the ra and dec of the halo such that they are centered in (0,0)
        ra -= ramid.value
        dec -= decmid.value

        ra,dec  = da.deg2rad(ra),da.deg2rad(dec)
        if self.cube_mode == 'flat_sky':
            rmid = ((self.cosmo.comoving_radial_distance(self.zmid)*u.Mpc).to(self.Mpch)).value
            # cartesian coordinates in flat sky
            x = da.ones(ra.shape[0])
            y = da.sin(ra)/r*rmid # only ra?
            z = da.sin(dec)/r*rmid # only dec?
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
                    
        # ~ ra,dec  = da.deg2rad(ra),da.deg2rad(dec)
        # ~ # cartesian coordinates
        # ~ x = da.cos(dec) * da.cos(ra)
        # ~ y = da.cos(dec) * da.sin(ra)
        # ~ z = da.sin(dec)
        # ~ pos = da.vstack([x,y,z]).T
        # ~ #radial distances in Mpch/h
        # ~ r = redshift.map_blocks(lambda zz: (((self.cosmo.comoving_radial_distance(zz)*u.Mpc).to(self.Mpch)).value),dtype=redshift.dtype)
        
        pos = da.vstack([x,y,z]).T
        cartesian_pixelpos = r[:,None] * pos
        foreground_grid = np.array(cartesian_pixelpos.compute())
        
        if self.do_inner_cut and not self.do_flat_sky:
            filtering = (foreground_grid[:,1] > raside_lim[0]) & (foreground_grid[:,1] < raside_lim[1]) & \
                        (foreground_grid[:,2] > decside_lim[0]) & (foreground_grid[:,2] < decside_lim[1])
            foreground_grid = foreground_grid[filtering]
            #print(np.asarray(filtering).shape, np.asarray(foreground_grid).shape, np.asarray(foreground_signal).shape)
            foreground_signal=np.asarray(foreground_signal).flatten()[filtering]
        #print(np.min(foreground_grid[:,0]), np.max(foreground_grid[:,0]), np.min(foreground_grid[:,1]), np.max(foreground_grid[:,1]), np.min(foreground_grid[:,2]), np.max(foreground_grid[:,2]))
        for n in range(3):
            foreground_grid[:,n] -= mins[n]
        #Set the emitter in the grid and paint using pmesh directly instead of nbk
        pm = pmesh.pm.ParticleMesh(Nmesh, BoxSize=Lbox, dtype='float32', resampler='cic')
        #Make realfield object

        field = pm.create(type='real')
        layout = pm.decompose(foreground_grid)
        #Exchange positions between different MPI ranks
        p = layout.exchange(foreground_grid)
        #Assign weights following the layout of particles
        m = layout.exchange(np.asarray(foreground_signal).flatten())
        pm.paint(p, out=field, mass=m, resampler='cic')
        #Fourier transform fields and apply the filter
        field = field.r2c()
        return field


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
