'''
Base module to pain the LIM lightcone
'''

import numpy as np
from glob import glob
from astropy.io import fits
from scipy.interpolate import interp2d

import os
import camb
import astropy.units as u
import astropy.constants as cu

import source.line_models as LM
import source.external_sfrs as extSFRs

from source.utilities import check_params,get_default_params
from source.utilities import cached_lightcone_property,cached_read_property, check_updated_params

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
                            
    -LIR_pars               Dictionary with the parameters required to compute infrared
                            luminosity, needed to compute certain lines luminosities.
                            Check the LIR function in line_models for the required parameters 
                            and available models

    -do_external_SFR        Boolean, whether to use a SFR different than Universe Machine
                            (default:False)

    -external_SFR           SFR interpolation

    -sig_extSFR             log-scatter for an external SFR
   
    -seed                   seed for the RNG object
    '''
    def __init__(self,
                 halo_lightcone_dir = '',
                 zmin = 0., zmax = 20.,
                 RA_min = -65.*u.deg,RA_max = 60.*u.deg,
                 DEC_min = -1.25*u.deg,DEC_max = 1.25*u.deg,
                 lines = dict(CO_J10 = False, CII = False, Halpha = False, Hbeta = False, Lyalpha = False, HI = False, 
                              CO_J21 = False, CO_J32 = False, CO_J43 = False, CO_J54 = False, CO_J65 = False, CO_J76 = False,
                              NIII = False, NII = False, OIII_88 = False, OI_63 = False, OI_145 = False, OII = False, OIII_0p5 = False),
                 models = dict(CO_J10 = dict(model_name = '', model_pars = {}), CII = dict(model_name = '', model_pars = {}), 
                               Halpha = dict(model_name = '', model_pars = {}), Hbeta = dict(model_name = '', model_pars = {}), 
                               Lyalpha = dict(model_name = '', model_pars = {}), HI = dict(model_name = '', model_pars = {}), 
                               CO_J21 = dict(model_name = '', model_pars = {}), CO_J32 = dict(model_name = '', model_pars = {}), 
                               CO_J43 = dict(model_name = '', model_pars = {}), CO_J54 = dict(model_name = '', model_pars = {}), 
                               CO_J65 = dict(model_name = '', model_pars = {}), CO_J76 = dict(model_name = '', model_pars = {}), 
                               NIII = dict(model_name = '', model_pars = {}), NII = dict(model_name = '', model_pars = {}), 
                               OIII_88 = dict(model_name = '', model_pars = {}), OI_63 = dict(model_name = '', model_pars = {}), 
                               OI_145 = dict(model_name = '', model_pars = {}), OII = dict(model_name = '', model_pars = {}), OIII_0p5 = dict(model_name = '', model_pars = {})),
                 LIR_pars = {},
                 do_external_SFR = False, external_SFR = '',sig_extSFR = 0.3, SFR_pars=dict(M0=1e-6, Ma=10**8, Mb=10**12.3, a=1.9, b=3.0, c=-1.4), 
                 seed=None):

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
        self._update_read_list = []
        self._update_survey_list = []
        self._update_measure_list = []

        #Initialize camb (we need background only) - values used in UM
        camb_pars = camb.set_params(H0=67.8, omch2 = 0.118002988, ombh2 = 0.02312)
        self.h = 0.678
        self.cosmo = camb.get_background(camb_pars)

        #Line frequencies:
        self.line_nu0 = getattr(LM,'lines_included')(self)
        
        self.rng = np.random.default_rng(self.seed)

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

    @cached_read_property
    def halo_catalog(self):
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
        dist_edges = (np.arange(Nfiles+1)+ind[[sort_ind[0]]])*25*self.Mpch.value
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

        return bigcat

    @cached_lightcone_property
    def L_line_halo(self):
        '''
        Computes the halo luminosity for each of the lines of interest
        '''
        L_line_halo = {}
        nuObs_line_halo = {}
        #Get the SFR
        if self.do_external_SFR:
            if self.external_SFR == 'Custom_SFR' or self.external_SFR == 'Dongwoo_SFR':
                #convert halo mass to Msun
                Mhalo_Msun = (self.halo_catalog['M_HALO']*self.Msunh).to(u.Msun)
                SFR = getattr(extSFRs,self.external_SFR)(Mhalo_Msun.value,self.halo_catalog['Z'], self.SFR_pars)
                #Add scatter to the relation
                sigma_base_e = self.sig_extSFR*2.302585
                SFR = SFR*self.rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, SFR.shape)
#            elif self.external_SFR == 'Behroozi_SFR':
#                #Build interpolating table at this step.
#                SFR_folder = os.path.dirname(os.path.realpath(__file__)).split("source")[0]+'SFR_tables/'
#                SFR_file = 'sfr_table_Behroozi.dat'
#
#                try:
#                    x = np.loadtxt(SFR_folder+SFR_file)
#                except:
#                    x = np.loadtxt(SFR_file)
#                zb = np.unique(x[:,0])-1.
#                logMb = np.unique(x[:,1])
#                logSFRb = x[:,2].reshape(len(zb),len(logMb),order='F')
#
#                logSFR_interp = interp2d(logMb,zb,logSFRb,bounds_error=False,fill_value=-40.)
#                logM = np.log10((self.halo_catalog['M_HALO']))
#                z = self.halo_catalog['Z']
#                if np.array(z).size>1:
#                    SFR = np.zeros(logM.size)
#                    for ii in range(0,logM.size):
#                        SFR[ii] = 10.**logSFR_interp(logM[ii],z[ii])
#                else:
#                    SFR = 10.**logSFR_interp(logM,z)
#
#                sigma_base_e = self.sig_extSFR*2.302585
#                SFR = SFR*self.rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, SFR.shape)

            else:
                #convert halo mass to Msun/h
                if self.external_SFR == 'Behroozi_SFR':
                    Mhalo_Msun = (self.halo_catalog['M_HALO']*self.Msunh)
                else:
                    Mhalo_Msun = (self.halo_catalog['M_HALO']*self.Msunh).to(u.Msun)
                SFR = getattr(extSFRs,self.external_SFR)(Mhalo_Msun.value,self.halo_catalog['Z'])
                sigma_base_e = self.sig_extSFR*2.302585
                SFR = SFR*self.rng.lognormal(-0.5*sigma_base_e**2, sigma_base_e, SFR.shape)
        else:
            SFR = self.halo_catalog['SFR_HALO']
            
        if len(self.LIR_pars.keys())>0:
            LIR = getattr(LM,'LIR')(self,SFR,self.LIR_pars,self.rng)
        else:
            LIR = 0*u.Lsun


        for line in self.lines.keys():
            if self.lines[line]:
                L_line_halo[line] = getattr(LM,self.models[line]['model_name'])(self,SFR,LIR,self.models[line]['model_pars'],self.line_nu0[line],self.rng)

        return L_line_halo

    @cached_lightcone_property
    def nuObs_line_halo(self):
        '''
        Computes the observed frequency for each halo and line
        '''
        nuObs_line_halo = {}

        for line in self.lines.keys():
            if self.lines[line]:
                nuObs_line_halo[line] = self.line_nu0[line]/(1+self.halo_catalog['Z']+self.halo_catalog['DZ'])

        return nuObs_line_halo

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
        read_params = ['halo_lightcone_dir', 'zmin', 'zmax',
                       'RA_min', 'RA_max', 'DEC_min', 'DEC_max']
        for name in read_params:
            lightcone_params.remove(name)
            
        if any(item in read_params for item in new_params.keys()):
            for attribute in self._update_read_list:
                delattr(self,attribute)
            self._update_read_list = []
            for attribute in self._update_lightcone_list:
                delattr(self,attribute)
            self._update_lightcone_list = []
            for attribute in self._update_survey_list:
                delattr(self,attribute)
            self._update_survey_list = []
            for attribute in self._update_measure_list:
                delattr(self,attribute)
            self._update_measure_list = []
        elif any(item in lightcone_params for item in new_params.keys()):
            for attribute in self._update_lightcone_list:
                delattr(self,attribute)
            self._update_lightcone_list = []
            for attribute in self._update_survey_list:
                delattr(self,attribute)
            self._update_survey_list = []
            for attribute in self._update_measure_list:
                delattr(self,attribute)
            self._update_measure_list = []
        elif any(item in survey_params for item in new_params.keys()):
            for attribute in self._update_survey_list:
                delattr(self,attribute)
            self._update_survey_list = []
            for attribute in self._update_measure_list:
                delattr(self,attribute)
            self._update_measure_list = []
        elif any (item in measure_params for item in new_params.keys()):
            for attribute in self._update_measure_list:
                delattr(self,attribute)
            self._update_measure_list = []
        #update parameters
        for key in new_params:
            setattr(self, key, new_params[key])

        #Update calls should maybe reset RNG for reproducibility?    
        self.rng = np.random.default_rng(self.seed)
        #check updated paramters:
        check_updated_params(self)

        return
