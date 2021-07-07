'''
Base module to pain the LIM lightcone
'''

import numpy as np
from glob import glob
from astropy.io import fits

import camb
import astropy.units as u
import astropy.constants as cu

import source.line_models as LM
import source.external_sfrs as extSFRs

from source.utilities import check_params,get_default_params
from source.utilities import cached_lightcone_property,cached_survey_property, check_updated_params

class Lightcone(object):
    '''
    An object controlling all relevant quantities needed to create the
    painted LIM lightcone. It reads a lightcone catalog of halos with SFR
    quantities and paint it with as many lines as desired.

    Allows to compute summary statistics as power spectrum and the VID for
    the signal (i.e., without including observational effects).

    Lines included: CO

    INPUT PARAMETERS:
    ------------------

    -halo_lightcone_dir     Path to the directory containing all files related to
                            the halo lightcone catalog

    -zmin,zmax              Minimum and maximum redshifts to read from the lightcone
                            (default: 0,20 - limited by Universe Machine)

    -RA_min,RA_max:         minimum and maximum RA to read from the lightcone
                            (Default = -65-60 deg)

    -DEC_min,DEC_max:       minimum and maximum DEC to read from the lightcone
                            (Default = -1.25-1.25 deg)

    -lines                  What lines are painted in the lightcone. Dictionary with
                            bool values (default: All false).
                            Available lines: CO, CII, H-alpha, Lyman-alpha, HI

    -models                 Models for each line. Dictionary of dictionaries (first layer,
                            same components of "lines", second layer, the following
                            components: model_name, model_pars (depends on the model))
                            (default: empty dictionary)

    -do_external_SFR        Boolean, whether to use a SFR different than Universe Machine
                            (default:False)

    -external_SFR           SFR interpolation

    -sig_extSFR             log-scatter for an external SFR
    '''
    def __init__(self,
                 halo_lightcone_dir = '',
                 zmin = 0., zmax = 20.,
                 RA_min = -65.*u.deg,RA_max = 60.*u.deg,
                 DEC_min = -1.25*u.deg,DEC_max = 1.25*u.deg,
                 lines = dict(CO = False, CII = False, Halpha = False, Lyalpha = False, HI = False),
                 models = dict(CO = dict(model_name = '', model_pars = {}), CII = dict(model_name = '', model_pars = {}),
                               Halpha = dict(model_name = '', model_pars = {}), Lyalpha = dict(model_name = '', model_pars = {}),
                               HI = dict(model_name = '', model_pars = {})),
                 do_external_SFR = False, external_SFR = '',sig_extSFR = 0.3, SFR_pars=dict(M0=1e-6, Ma=10**8, Mb=10**12.3, a=1.9, b=3.0, c=-1.4)):

        # Get list of input values to check type and units
        self._lightcone_params = locals()
        self._lightcone_params.pop('self')

        # Get list of input names and default values
        self._default_lightcone_params = get_default_params(Lightcone.__init__)
        # Check that input values have the correct type and units
        check_params(self._lightcone_params,self._default_lightcone_params)
        # Fill lines no included with false
        for key in list(self._default_lightcone_params['lines'].keys()):
            if key not in self._lightcone_params['lines'].keys():
                self._lightcone_params['lines'][key] = False

        # Set all given parameters
        for key in self._lightcone_params:
            setattr(self,key,self._lightcone_params[key])

        # Create overall lists of parameters (Only used if using one of
        self._input_params = {}
        self._default_params = {}
        self._input_params.update(self._lightcone_params)
        self._default_params.update(self._default_lightcone_params)

        # Create list of cached properties
        self._update_lightcone_list = []
        self._update_survey_list = []
        self._update_measure_list = []

        #Placeholder
        self.L_line_halo = None

        #Initialize camb (we need background only) - values used in UM
        camb_pars = camb.set_params(H0=67.8, omch2 = 0.118002988, ombh2 = 0.02312)
        self.h = 0.678
        self.cosmo = camb.get_background(camb_pars)

        #Line frequencies:
        self.line_nu0 = dict(CO = 115.271*u.GHz, CII = 1900.539*u.GHz, HI = 1.4204134*u.GHz,
                        Lyalpha = 2465398.5*u.GHz, Halpha = 456805.72*u.GHz, Hbeta = 616730.01028595*u.GHz,
                        OII = 804380.08585994*u.GHz, OIII = 598746.67066107*u.GHz)

    #########
    # Units #
    #########

    @cached_lightcone_property
    def Mpch(self):
        '''
        Mpc/h unit, required for interacting with hmf outputs
        '''
        return u.Mpc/self.h

    @cached_lightcone_property
    def Msunh(self):
        '''
        Msun/h unit, required for interacting with hmf outputs
        '''
        return u.Msun/self.h

    ######################
    # Catalog Management #
    ######################

    @cached_lightcone_property
    def read_halo_catalog(self):
        '''
        Reads all the files from the halo catalog and appends the slices
        '''
        fnames = glob(self.halo_lightcone_dir+'/*')
        Nfiles = len(fnames)
        #get the sorted indices from fnames
        ind = np.zeros(Nfiles).astype(int)
        for ifile in range(Nfiles):
            ind[ifile] =  int(fnames[ifile].split('_')[-1].split('.')[0])
        sort_ind = np.argsort(ind)
        #get the edge distances for each slice in Mpc (25 Mpc/h width each slice)
        dist_edges = (np.arange(Nfiles+1))*25*self.Mpch.value
        min_dist = self.cosmo.comoving_radial_distance(self.zmin)
        max_dist = self.cosmo.comoving_radial_distance(self.zmax)
        inds_in = np.where(np.logical_and(dist_edges[:-1] >= min_dist, dist_edges[1:] <= max_dist))[0]
        N_in = len(inds_in)
        #open the first one
        fil = fits.open(fnames[sort_ind[inds_in[0]]])
        print(fnames[sort_ind[inds_in[0]]])

        #Start the catalog appending everything
        bigcat = np.array(fil[1].data)
        #Open the rest and append
        for ifile in range(1,N_in):
            print(fnames[sort_ind[inds_in[ifile]]])
            fil = fits.open(fnames[sort_ind[inds_in[ifile]]])
            data = np.array(fil[1].data)
            inds_RA = (data['RA'] > self.RA_min.value)&(data['RA'] < self.RA_max.value)
            inds_DEC = (data['DEC'] > self.DEC_min.value)&(data['DEC'] < self.DEC_max.value)
            inds_sky = inds_RA&inds_DEC
            bigcat = np.append(bigcat, data[inds_sky])

        self.halo_catalog = bigcat
        return

    @cached_lightcone_property
    def halo_luminosity(self):
        '''
        Computes the halo luminosity for each of the lines of interest,
        and the corresponding observed frequency for each halo and line
        '''
        L_line_halo = {}
        nuObs_line_halo = {}
        #Get the SFR
        if self.do_external_SFR:
            #convert halo mass to Msun
            Mhalo_Msun = (self.halo_catalog['M_HALO']*self.Msunh).to(u.Msun)
            if self.external_SFR == 'Custom_SFR':
                SFR = getattr(extSFRs,self.external_SFR)(Mhalo_Msun.value,self.halo_catalog['Z'], self.SFR_pars)
                #Add scatter to the relation
                sigma_base_e = sig_extSFR*2.302585
                SFR = SFR*np.random.lognormal(-0.5*sigma_base_e**2, sigma_base_e, SFR.shape)
            else:
                SFR = getattr(extSFRs,self.external_SFR)(Mhalo_Msun.value,self.halo_catalog['Z'])
                sigma_base_e = sig_extSFR*2.302585
                SFR = SFR*np.random.lognormal(-0.5*sigma_base_e**2, sigma_base_e, SFR.shape)
        else:
            SFR = self.halo_catalog['SFR_HALO']


        for line in self.lines.keys():
            if self.lines[line]:
                L_line_halo[line] = getattr(LM,self.models[line]['model_name'])(self,SFR,self.models[line]['model_pars'])
                nuObs_line_halo[line] = self.line_nu0[line]/(1+self.halo_catalog['Z']+self.halo_catalog['DZ'])

        self.L_line_halo = L_line_halo
        self.nuObs_line_halo = nuObs_line_halo

        return

    def make_lightcone(self):
        '''
        Wrapper for "read_halo_catalog" and "halo_luminosity"
        '''
        self.read_halo_catalog
        self.halo_luminosity
        return

    def save_lightcone(self):
        '''
        Saves the lightcone in a fits file
        '''
        #DO WE WANT TO DO THIS OR ONLY AFTER MAKE SURVEY??? MAYBE TOO LARGE?

        #IF NOT, REMOVE OUTPUT_ROOT
        return

    ########################################################################
    # Method for updating input parameters and resetting cached properties #
    ########################################################################
    def update(self, **new_params):
        # Check if params dict contains valid parameters
        check_params(new_params,self._default_params)
        #update the class that corresponds
        lightcone_params = list(self._default_lightcone_params.keys())
        survey_params = list(self._default_survey_params.keys())
        measure_params = list(self._default_measure_params.keys())
        if any(item in lightcone_params for item in new_params.keys()):
            for attribute in self._update_lightcone_list:
                delattr(self,attribute)
            self._update_lightcone_list = []
            for attribute in self._update_survey_list:
                delattr(self,attribute)
            self._update_survey_list = []
            for attribute in self._update_measure_list:
                delattr(self,attribute)
            self._update_measure_list = []
        if any(item in survey_params for item in new_params.keys()):
            for attribute in self._update_survey_list:
                delattr(self,attribute)
            self._update_survey_list = []
            for attribute in self._update_measure_list:
                delattr(self,attribute)
            self._update_measure_list = []
        if any (item in measure_params for item in new_params.keys()):
            for attribute in self._update_measure_list:
                delattr(self,attribute)
            self._update_measure_list = []
        #update parameters
        for key in new_params:
            setattr(self, key, new_params[key])
            
        #check updated paramters:
        check_updated_params(self)

        return
