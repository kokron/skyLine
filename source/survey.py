'''
Base module to make a LIM survey from painted lightcone
'''

import numpy as np
from scipy.interpolate import interp2d,interp1d
from scipy.special import legendre
import dask.array as da
import astropy.units as u
import astropy.constants as cu
import copy
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.algorithms import FFTPower
from nbodykit.source.mesh.catalog import CompensateCICShotnoise
import pmesh
from pmesh.pm import RealField, ComplexField
from source.lightcone import Lightcone
from source.utilities import cached_survey_property,get_default_params,check_params
from source.utilities import set_lim, dict_lines

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

    -dk:                    k spacing for the power spectrum (default: 0.02 Mpc^-1~0.01 h/Mpc)

    -kmin,kmax:             Minimum and maximum k values for the power spectrum
                            (default: 0., 3 Mpc^-1 ~ 5 h/Mpc)

    -Nmu:                   Number of sampling in mu to compute the power spectrum
                            (default: 10)

    -remove_noise:          Remove the expected instrumental noise power spectrum (sigma_N^2*Vvox)
                            from the observed power spectrum (and adds it to the covariance).
                            (default: False)

    -linear_VID_bin:        Boolean, to do linear (or log) binning for the VID histogram
                            (default: False)

    -paint_catalog:         Boolean: Paint catalog or used a painted one.               DOES THIS MAKE SENSE OR ALWAYS TRUE????
                            (Default: True).

    -do_smooth:             Boolean: apply smoothing filter to implement resolution
                            limitations. (Default: True)

    -output_root            Root path for output products. (default: output/default)
    
    -do_inner_cut           Get a box for which there are no empty spaces, but discards some haloes.
                            (Default: True). Do this *only* for narrow fields
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
                 dk = 0.02*u.Mpc**-1,
                 kmin = 0.0*u.Mpc**-1,
                 kmax = 3.*u.Mpc**-1,
                 Nmu = 5,
                 remove_noise = False,
                 output_root = "output/default",
                 paint_catalog = True,
                 do_smooth = True,
                 do_inner_cut = True,
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
               
        # Check that the bandwidth and lines used are included in the lightcone limits
        for line in self.lines.keys():
            if self.lines[line]:
                #Get true cell volume
                zlims = (self.line_nu0[line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
                if zlims[0] <= self.zmin or zlims [1] >= self.zmax:
                    raise ValueError('The line {} on the bandwidth [{},{}] corresponds to z range [{},{}], while the included redshifts in the lightcone are within [{},{}]. Please remove the line, increase the zmin,zmax range or reduce the bandwith.'.format(line,self.nuObs_max,self.nuObs_min,zlims[0],zlims[1],zmin,zmax))
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

    @cached_survey_property
    def Lbox(self):
        '''
        Sides of the field observed (approximated to be a rectangular cube),
        for the assumed redshift (the one corresponding to the target line)
        '''
        #box angular limits
        ralim = np.deg2rad(np.array([self.RAObs_min.value,self.RAObs_max.value]))
        declim = np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value]))
        #transform Frequency band into redshift range for the target line
        zlims = (self.line_nu0[self.target_line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
        rlim = ((self.cosmo.comoving_radial_distance(zlims)*u.Mpc).to(self.Mpch)).value
        #Get the side of the box
        if self.do_inner_cut:
            raside = 2*rlim[0]*np.sin(0.5*(ralim[1]-ralim[0]))
            decside = 2*rlim[0]*np.sin(0.5*(declim[1]-declim[0]))
            zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
        else:
            raside = 2*rlim[1]*np.sin(0.5*(ralim[1]-ralim[0]))
            decside = 2*rlim[1]*np.sin(0.5*(declim[1]-declim[0]))
            zside = rlim[1]-rlim[0]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))
        Lbox = np.array([zside,raside,decside])
            
        self.raside_lim = rlim[0]*np.sin(ralim) #min, max
        self.decside_lim = rlim[0]*np.sin(declim) #min, max

        return (Lbox*self.Mpch).to(self.Mpch)

    @cached_survey_property
    def Vvox(self):
        '''
        Voxel volume (approximated as a rectangular cube) for the assumed redshift
        (the one corresponding to the target line).

        Voxel dimensions given by the resolution of the experiment.
        '''
        Nmesh = np.array([self.Nchan,self.Nside[0], self.Nside[1]], dtype=int)
        #return (self.Lbox.value/Nmesh).prod()*self.Lbox.unit**3
        return (self.Lbox.value/Nmesh).prod()*self.Lbox.unit**3

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
                halos_survey[line]['Ztrue'] = np.append(halos_survey[line]['Ztrue'],self.halo_catalog['Z'][inds]+self.halo_catalog['DZ'][inds])
                halos_survey[line]['Lhalo'] = np.append(halos_survey[line]['Lhalo'],self.L_line_halo[line][inds])

        return halos_survey

    @cached_survey_property
    def obs_fourier_map(self):
        '''
        Generates the mock intensity map observed in Fourier space,
        obtained from Cartesian coordinates.

        Use obs_fourier_map.c2r() to get the real field
        '''
        #Define the mesh divisions and the box size
        Nmesh = np.array([self.supersample*self.Nchan,
                  self.supersample*self.Nside[0],
                  self.supersample*self.Nside[1]], dtype=int)
        Lbox = self.Lbox.value
        
        ralim = np.deg2rad(np.array([self.RAObs_min.value,self.RAObs_max.value]))
        declim = np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value]))
        raside_lim = self.raside_lim
        decside_lim = self.decside_lim
        
        global sigma_par
        global sigma_perp
        maps = np.zeros([Nmesh[0],Nmesh[1],Nmesh[2]//2 + 1],dtype='complex64')
        
        for line in self.lines.keys():
            if self.lines[line]:
                #Get true cell volume
                zlims = (self.line_nu0[line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
                rlim = ((self.cosmo.comoving_radial_distance(zlims)*u.Mpc).to(self.Mpch)).value
                #Get the side of the box
                if self.do_inner_cut:
                    raside = 2*rlim[0]*np.sin(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rlim[0]*np.sin(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
                    rside_lim = np.array([rlim[0],rlim[0]+zside])
                else:
                    raside = 2*rlim[1]*np.sin(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rlim[1]*np.sin(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]-rlim[0]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))
                    rside_lim = np.array([rlim[1]-zside,rlim[1]])
                Lbox_true = np.array([zside,raside,decside])
                Vcell_true = (Lbox_true/Nmesh).prod()*(self.Mpch**3).to(self.Mpch**3)
                #Get positions using the observed redshift
                #Convert the halo position in each volume to Cartesian coordinates (from Nbodykit)
                ra,dec,redshift = da.broadcast_arrays(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'],
                                                      self.halos_in_survey[line]['Zobs'])
                ra,dec  = da.deg2rad(ra),da.deg2rad(dec)
                # cartesian coordinates
                x = da.cos(dec) * da.cos(ra)
                y = da.cos(dec) * da.sin(ra)
                z = da.sin(dec)
                pos = da.vstack([x,y,z]).T
                #radial distances in Mpch/h
                r = redshift.map_blocks(lambda zz: (((self.cosmo.comoving_radial_distance(zz)*u.Mpc).to(self.Mpch)).value),
                                        dtype=redshift.dtype)
                cartesian_halopos = r[:,None] * pos
                lategrid = np.array(cartesian_halopos.compute())
                #Filter some halos out if outside of the inner cut
                if self.do_inner_cut:
                    filtering = (lategrid[:,0] >= rside_lim[0]) & (lategrid[:,0] <= rside_lim[1]) & \
                                (lategrid[:,1] >= raside_lim[0]) & (lategrid[:,1] <= raside_lim[1]) & \
                                (lategrid[:,2] >= decside_lim[0]) & (lategrid[:,2] <= decside_lim[1])
                    lategrid = lategrid[filtering]
                    #Compute the signal in each voxel (with Ztrue and Vcell_true)
                    Zhalo = self.halos_in_survey[line]['Ztrue'][filtering]
                    Hubble = self.cosmo.hubble_parameter(Zhalo)*(u.km/u.Mpc/u.s)
                    if self.do_intensity:
                        #intensity[Jy/sr]
                        signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo'][filtering]/Vcell_true).to(self.unit)
                    else:
                        #Temperature[uK]
                        signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo'][filtering]/Vcell_true).to(self.unit)
                else:
                    Zhalo = self.halos_in_survey[line]['Ztrue']
                    Hubble = self.cosmo.hubble_parameter(Zhalo)*(u.km/u.Mpc/u.s)
                    if self.do_intensity:
                        #intensity[Jy/sr]
                        signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                    else:
                        #Temperature[uK]
                        signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                #Locate the grid such that bottom left corner of the box is [0,0,0] which is the nbodykit convention.
                for n in range(3):
                    lategrid[:,n] -= np.min(lategrid[:,n])
                #Set the emitter in the grid and paint using pmesh directly instead of nbk
                pm = pmesh.pm.ParticleMesh(Nmesh, BoxSize=Lbox, dtype='float32', resampler='cic')
                #Make realfield object
                field = pm.create(type='real')
                layout = pm.decompose(lategrid)
                #Exchange positions between different MPI ranks
                p = layout.exchange(lategrid)
                #Assign weights following the layout of particles
                m = layout.exchange(signal.value)
                pm.paint(p, out=field, mass=m, resampler='cic')
                #Fourier transform fields and apply the filter
                field = field.r2c()
                #Compensate the field for the CIC window function we apply
                field = field.apply(CompensateCICShotnoise, kind='circular')
                #This smoothing comes from the resolution window function.
                if self.do_smooth:
                    #compute scales for the anisotropic filter (in Ztrue -> zmid)
                    zmid = (self.line_nu0[line]/self.nuObs_mean).decompose().value-1
                    sigma_par = (cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
                    sigma_perp = (self.cosmo.comoving_radial_distance(zmid)*u.Mpc*(self.beam_width/(1*u.rad))).to(self.Mpch).value
                    field = field.apply(aniso_filter, kind='wavenumber')
                #Add this contribution to the total maps
                maps+=field

        #Add noise in the cosmic volume probed by target line
        if self.Tsys.value > 0.:
            #get the proper shape for the observed map
            if self.supersample > 1:
                pm_noise = pmesh.pm.ParticleMesh(np.array([self.Nchan,self.Nside[0],self.Nside[1]], dtype=int),
                                                  BoxSize=Lbox, dtype='float32', resampler='cic')
                maps = pm_noise.downsample(maps.c2r(),keep_mean=True)
            else:
                maps = maps.c2r()
            #distribution is positive gaussian with 0 mean
            #add the noise
            maps += np.random.normal(0.,self.sigmaN.value,maps.shape)

            return maps.r2c()
        else:
            return maps


################################
## Compute summary statistics ##
################################

    @cached_survey_property
    def Pk_2d(self):
       '''
       Computes the 2d power spectrum P(k,mu) of the map
       '''
       return FFTPower(self.obs_fourier_map, '2d', Nmu=self.Nmu, poles=[0,2,4], los=[1,0,0],
                       dk=self.dk.to(self.Mpch**-1).value,kmin=self.kmin.to(self.Mpch**-1).value,
                       kmax=self.kmax.to(self.Mpch**-1).value,BoxSize=self.Lbox.value)

    @cached_survey_property
    def k_Pk_poles(self):
        '''
        Fourier wavenumbers for the multipoles of the power spectrum
        '''
        return self.Pk_2d.poles['k']*self.Mpch**-1

    @cached_survey_property
    def Pk_0(self):
        '''
        Monopole of the power spectrum
        '''
        if self.remove_noise:
            return self.Pk_2d.poles['power_0'].real*self.Mpch**3*self.unit**2 - self.sigmaN**2*self.Vvox
        else:
            return self.Pk_2d.poles['power_0'].real*self.Mpch**3*self.unit**2

    @cached_survey_property
    def Pk_2(self):
        '''
        Quadrupole of the power spectrum
        '''
        return self.Pk_2d.poles['power_2'].real*self.Mpch**3*self.unit**2

    @cached_survey_property
    def Pk_4(self):
        '''
        Hexadecapole of the power spectrum
        '''
        return self.Pk_2d.poles['power_4'].real*self.Mpch**3*self.unit**2

    @cached_survey_property
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
        if 'sigma_LCO' in self.models[self.target_line]['model_pars']:
            sigma_scatter = self.models[self.target_line]['model_pars']['sigma_LCO']
        #ANY OTHER CASE?
        else:
            sigma_scatter = 0.
        M.update(nu=self.line_nu0[self.target_line],model_name=line_model,model_par=line_pars,
                      sigma_scatter = sigma_scatter)
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
                if 'sigma_LCO' in self.models[line]['model_pars']:
                    sigma_scatter = self.models[line]['model_pars']['sigma_LCO']
                #ANY OTHER CASE?
                else:
                    sigma_scatter = 0.
                M.update(nu=self.line_nu0[line],model_name=line_model,model_par=line_pars,
                              sigma_scatter = sigma_scatter)
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
                #Get the measured Pk contribution and add it to the rest
                PK_2d += interp2d(M.k,M.mu,M.Pk)(kprime,mu_prime)*PK_2d.unit

        return M.k.to(M.Mpch**-1),M.mu,PK_2d.to(self.Mpch**3*self.unit**2)

    @cached_survey_property
    def covmat_00(self):
        '''
        00 term of the total covariance matrix
        '''
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 0.5*np.trapz(integrand**2,self.Pk_2d_theo[1],axis=0)
        return interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles[1:])/self.Pk_2d.poles['modes'][1:]*self.Mpch**6*self.unit**4


    @cached_survey_property
    def covmat_02(self):
        '''
        02 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L2 = legendre(2)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 5./2.*np.trapz(integrand**2*L2,self.Pk_2d_theo[1],axis=0)
        return interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6*self.unit**4


    @cached_survey_property
    def covmat_04(self):
        '''
        04 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L4 = legendre(4)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 9./2.*np.trapz(integrand**2*L4,self.Pk_2d_theo[1],axis=0)
        return interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6*self.unit**4


    @cached_survey_property
    def covmat_22(self):
        '''
        22 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L2 = legendre(2)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 25./2.*np.trapz(integrand**2*L2*L2,self.Pk_2d_theo[1],axis=0)
        return interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6*self.unit**4


    @cached_survey_property
    def covmat_24(self):
        '''
        24 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L2 = legendre(2)(mui_grid)
        L4 = legendre(4)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 45./2.*np.trapz(integrand**2*L2*L4,self.Pk_2d_theo[1],axis=0)
        return interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6*self.unit**4


    @cached_survey_property
    def covmat_44(self):
        '''
        44 term of the total covariance matrix
        '''
        mui_grid = np.meshgrid(self.Pk_2d_theo[0],self.Pk_2d_theo[1])[1]
        L4 = legendre(4)(mui_grid)
        integrand = (self.Pk_2d_theo[2]+self.sigmaN**2*self.Vvox)
        cov = 81./2.*np.trapz(integrand**2*L4*L4,self.Pk_2d_theo[1],axis=0)
        return interp1d(self.Pk_2d_theo[0],cov)(self.k_Pk_poles)/self.Pk_2d.poles['modes']*self.Mpch**6*self.unit**4


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

    @cached_survey_property
    def Ti_edge(self):
        '''
        Edges of the VID histogram bins
        '''
        if self.linear_VID_bin:
            Te = np.linspace(self.Tmin_VID.value,self.Tmax_VID.value,self.Nbin_hist+1)*self.Tmin_VID.unit
        else:
            Te = np.logspace(np.log10(self.Tmin_VID.value),np.log10(self.Tmax_VID.value),self.Nbin_hist+1)*self.Tmin_VID.unit
        return Te

    @cached_survey_property
    def Ti(self):
        '''
        Center of the VID histogram bins
        '''
        return (self.Ti_edge[:-1]+self.Ti_edge[1:])/2.

    @cached_survey_property
    def Bi_VID(self):
        '''
        Computes the histogram of temperatures in each voxel in hte observed map.
        Equivalent to the VID
        '''
        return np.histogram(np.array(self.obs_fourier_map.c2r()).flatten(),
                            bins=self.Ti_edge.value)[0]

    @cached_survey_property
    def Bi_VID_covariance(self):
        '''
        Covariance matrix of the VID histograms
        '''
        return np.diag(self.Bi_VID)




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
