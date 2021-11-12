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
from nbodykit.source.mesh.catalog import CompensateCICShotnoise
import healpy as hp
from source.lightcone import Lightcone
from source.utilities import cached_survey_property,get_default_params,check_params
from source.utilities import set_lim, dict_lines



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

    -supersample:           Factor of supersample with respect to the survey resolution
                            when making the grid. (Default: 10)

    -paint_catalog:         Boolean: Paint catalog or used a painted one.               DOES THIS MAKE SENSE OR ALWAYS TRUE????
                            (Default: True).

    -do_smooth:             Boolean: apply smoothing filter to implement resolution
                            limitations. (Default: True)

    -do_inner_cut           Get a box for which there are no empty spaces, but discards some haloes.
                            (Default: True). Do this *only* for narrow fields

    -do_angular             Create an angular survey (healpy map)
                            (Default: False)
                            
    -average_angular_proj   Average total integrated intensity per the number of channels for
                            angular projections (Default: True)
                            
    -nside                  NSIDE used by healpy to create angular maps. (Default: 2048)
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
                 supersample = 10, 
                 do_smooth = True,
                 do_inner_cut = True,
                 do_angular = False,
                 average_angular_proj = True,
                 nside = 2048,
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
                zlims = (self.line_nu0[line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
                if zlims[0] <= self.zmin or zlims [1] >= self.zmax:
                    raise ValueError('The line {} on the bandwidth [{:.2f},{:.2f}] corresponds to z range [{:.2f},{:.2f}], while the included redshifts in the lightcone are within [{:.2f},{:.2f}]. Please remove the line, increase the zmin,zmax range or reduce the bandwith.'.format(line,self.nuObs_max,self.nuObs_min,zlims[0],zlims[1],self.zmin,self.zmax))

        #Check healpy pixel size just in case:
        if self.do_angular:
            if (self.beam_FWHM.to(u.arcmin)).value < hp.nside2resol(self.nside, arcmin=True):
                print("WARNING!!! the healpy pixel side chosen, from NSIDE = {}, is {:.2f} times bigger than the beam_FWHM. Consider increasing NSIDE (remember that it must be a power of 2)".format(self.nside,hp.nside2resol(self.nside, arcmin=True)/self.beam_FWHM.to(u.arcmin)).value)
            #Avoid inner cut if do_angular:
            if self.do_angular and self.do_inner_cut:
                raise ValueError('If you want to work with angular maps, you do not need the inner cut, hence please use do_inner_cut = False')

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
            raside = 2*rlim[0]*np.tan(0.5*(ralim[1]-ralim[0]))
            decside = 2*rlim[0]*np.tan(0.5*(declim[1]-declim[0]))
            zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
            rside_lim = np.array([rlim[0],rlim[0]+zside])
        else:
            raside = 2*rlim[1]*np.tan(0.5*(ralim[1]-ralim[0]))
            decside = 2*rlim[1]*np.tan(0.5*(declim[1]-declim[0]))
            zside = rlim[1]-rlim[0]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))
            rside_lim = np.array([rlim[1]-zside,rlim[1]])
            
        Lbox = np.array([zside,raside,decside])

        self.raside_lim = rlim[0]*np.sin(ralim) #min, max
        self.decside_lim = rlim[0]*np.sin(declim) #min, max
        self.rside_obs_lim = rside_lim #min, max

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

        #Loop over lines to see what halos are within nuObs
        for line in self.lines.keys():
            if self.lines[line]:
                halos_survey[line] = dict(RA= np.array([]),DEC=np.array([]),Zobs=np.array([]),Ztrue=np.array([]),Lhalo=np.array([])*u.Lsun)
                #inds = (self.nuObs_line_halo[line] >= self.nuObs_min)&(self.nuObs_line_halo[line] <= self.nuObs_max)&inds_sky
                inds = (self.nuObs_line_halo[line] >= self.nuObs_min)&(self.nuObs_line_halo[line] <= self.nuObs_max)&inds_sky
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
            raise(Warning('Mask edges will be funky in this case, might see some vignetting'))
        npix = hp.nside2npix(self.nside)

        #This is too much memory
        # maps = np.zeros((self.Nchan, npix))

        hp_map = np.zeros(npix)

        # First, compute the intensity/temperature of each halo in the catalog we will include
        for line in self.lines.keys():
            if self.lines[line]:
                #Get true cell volume

                #Get positions using the observed redshift
                #Convert the halo position in each volume to Cartesian coordinates (from Nbodykit)
                ra,dec,redshift = da.broadcast_arrays(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'],
                                                      self.halos_in_survey[line]['Zobs'])


                Zhalo = self.halos_in_survey[line]['Ztrue']
                Hubble = self.cosmo.hubble_parameter(Zhalo)*(u.km/u.Mpc/u.s)


                #Figure out what channel the halos will be in to figure out the voxel volume, for the signal. 
                #This is what will be added to the healpy map.

                #Figure out which channel each line will end up in
                nu_bins = self.nuObs_min.to('GHz').value + np.arange(self.Nchan)*self.dnu.to('GHz').value 

                zmid_channel = nu_bins + 0.5*self.dnu.to('GHz').value

                #Channel of each halo, can now compute voxel volumes where each of them are seamlessly
                bin_idxs = np.digitize(self.line_nu0[line].to('GHz').value/(1+Zhalo), nu_bins)-1

                zmids = zmid_channel[bin_idxs]

                #Vcell = Omega_pix * D_A (z)^2 * (1+z) * Dnu/nu * c/H is the volume of the voxel for a given channel
                Vcell_true = hp.nside2pixarea(self.nside)*(self.cosmo.comoving_radial_distance(zmids)*u.Mpc )**2 * (1 + zmids) * (self.delta_nuObs/self.line_nu0[line]) * (cu.c.to('km/s')/Hubble)


                if self.do_intensity:
                    #intensity[Jy/sr]
                    signal = (cu.c/(4.*np.pi*self.line_nu0[line]*Hubble*(1.*u.sr))*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)
                else:
                    #Temperature[uK]
                    signal = (cu.c**3*(1+Zhalo)**2/(8*np.pi*cu.k_B*self.line_nu0[line]**3*Hubble)*self.halos_in_survey[line]['Lhalo']/Vcell_true).to(self.unit)


                #Paste the signals to the map
                theta, phi = rd2tp(self.halos_in_survey[line]['RA'], self.halos_in_survey[line]['DEC'])

                pixel_idxs = hp.ang2pix(self.nside, theta, phi)
                
                if self.average_angular_proj:
                    #averaging over the number of channels
                    np.add.at(hp_map, pixel_idxs, signal.value/self.Nchan)
                else:
                    np.add.at(hp_map, pixel_idxs, signal.value)
                #should smoothing be after masking?
                #could lead to bleeding of the zeros with the boundary
                if self.do_smooth:
                    theta_beam = self.beam_FWHM.to(u.rad)
                    hp_map = hp.smoothing(hp_map, theta_beam.value)

                #Define the mask from the rectangular footprint
                phicorner = np.deg2rad(np.array([self.RAObs_min.value,self.RAObs_min.value,self.RAObs_max.value,self.RAObs_max.value]))
                thetacorner = np.pi/2-np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value,self.DECObs_max.value,self.DECObs_min.value]))
                vecs = hp.dir2vec(thetacorner,phi=phicorner).T
                pix_within = hp.query_polygon(nside=self.nside,vertices=vecs,inclusive=False)
                self.pix_within = pix_within
                mask = np.ones(hp.nside2npix(self.nside),np.bool)
                mask[pix_within] = 0
                hp_map = hp.ma(hp_map)
                hp_map.mask = mask
                #add noise
        if self.Tsys.value > 0.:
            #rescale the noise per pixel to the healpy pixel size
            hp_sigmaN = self.sigmaN * (pix_within.size/self.Npix)**0.5
            hp_map[pix_within] += self.rng.normal(0.,hp_sigmaN.value,pix_within.size)
                    
        return hp_map


    @cached_survey_property
    def obs_3d_map(self):
        '''
        Generates the mock intensity map observed in Fourier space,
        obtained from Cartesian coordinates. It does not include noise.
        '''
        
        if self.do_angular:
            raise(Warning('Mask edges might be problematic due to the expanded selection!'))
        
        #Define the mesh divisions and the box size
        Nmesh = np.array([self.supersample*self.Nchan,
                  self.supersample*self.Nside[0],
                  self.supersample*self.Nside[1]], dtype=int)
        Lbox = self.Lbox.value

        ralim = np.deg2rad(np.array([self.RAObs_min.value,self.RAObs_max.value]))
        declim = np.deg2rad(np.array([self.DECObs_min.value,self.DECObs_max.value]))
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
                if self.do_inner_cut:
                    raside = 2*rlim[0]*np.tan(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rlim[0]*np.tan(0.5*(declim[1]-declim[0]))
                    zside = rlim[1]*np.cos(max(0.5*(ralim[1]-ralim[0]),0.5*(declim[1]-declim[0])))-rlim[0]
                    rside_lim = np.array([rlim[0],rlim[0]+zside])
                else:
                    raside = 2*rlim[1]*np.tan(0.5*(ralim[1]-ralim[0]))
                    decside = 2*rlim[1]*np.tan(0.5*(declim[1]-declim[0]))
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
                mins = np.array([rside_lim[0],raside_lim[0],decside_lim[0]])
                for n in range(3):
                    lategrid[:,n] -= mins[n]
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
                #This smoothing comes from the resolution window function.
                if self.do_smooth:
                    #compute scales for the anisotropic filter (in Ztrue -> zmid)
                    zmid = (self.line_nu0[line]/self.nuObs_mean).decompose().value-1
                    sigma_par = (cu.c*self.dnu*(1+zmid)/(self.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*self.nuObs_mean)).to(self.Mpch).value
                    sigma_perp = (self.cosmo.comoving_radial_distance(zmid)*u.Mpc*(self.beam_width/(1*u.rad))).to(self.Mpch).value
                    field = field.apply(aniso_filter, kind='wavenumber')
                #Add this contribution to the total maps
                maps+=field

        #Compensate the field for the CIC window function we apply
        #maps = maps.apply(CompensateCICShotnoise, kind='circular')

        #Add noise in the cosmic volume probed by target line to the 3d maps
        if self.Tsys.value > 0.:
            #Avoiding the downgrade of Nmesh to Nchan,Nside,Nside and rescaling sigmaN
            #TODO: check that this doesn't affect the VID SNR (signal will change since voxel size will change)
            
            #get the proper shape for the observed map
            #if self.supersample > 1:
            #    pm_noise = pmesh.pm.ParticleMesh(np.array([self.Nchan,self.Nside[0],self.Nside[1]], dtype=int),
            #                                          BoxSize=Lbox, dtype='float32', resampler='cic')
            #    maps = pm_noise.downsample(maps.c2r(),keep_mean=True)
            #    #Check if compensation is required again
            #    maps = (maps.r2c()).apply(CompensateCICShotnoise, kind='circular')

            #rescale noise:
            supersample_sigmaN = self.sigmaN * (self.supersample)**1.5
            maps = maps.c2r()
            #add the noise, distribution is gaussian with 0 mean
            maps += self.rng.normal(0.,supersample_sigmaN.value,maps.shape)
        else:
            maps = maps.c2r()
        return maps
            
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
