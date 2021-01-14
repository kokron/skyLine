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
from source.utilities import cached_lightcone_property

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
                            
    -lines                  What lines are painted in the lightcone. Dictionary with
                            bool values (default: All false). 
                            Available lines: CO, CII, H-alpha, Lyman-alpha, HI
                            
    -model                  Models for each line. Dictionary of dictionaries (first layer,
                            same components of "lines", second layer, the following 
                            components: model_name, model_pars (depends on the model))
                            (default: empty dictionary)
                            
    -do_external_SFR        Boolean, whether to use a SFR different than Universe Machine
                            (default:False)
                            
    -external_SFR           SFR interpolation
                            
    -output_root            Root path for output products. (default: output/default)                                
    '''
    def __init__(self,
                 halo_lightcone_dir = '',
                 zmin = 0., zmax = 20.,
                 lines = dict(CO = False, CII = False, Halpha = False, Lyalpha = False, HI = False),
                 models = dict(CO = dict(model_name = '', model_pars = {}), CII = dict(model_name = '', model_pars = {}),
                               Halpha = dict(model_name = '', model_pars = {}), Lyalpha = dict(model_name = '', model_pars = {}), 
                               HI = dict(model_name = '', model_pars = {})),
                 do_external_SFR = False, external_SFR = '',
                 output_root = "output/default"):
                 
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
        self._default_params.update(self._default_lighcone_params)
        
        # Create list of cached properties
        self._update_lightcone_list = []
        self._update_survey_list = []
            
        #Placeholder
        self.L_line_halo = None
        
        #Initialize camb (we need background only) - values used in UM
        camb_pars = camb.set_params(H0=68.0, omch2 = 0.1188368, ombh2 = 0.02312)
        self.h = 0.68
        self.cosmo = camb.get_background(camb_pars)
        
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
        min_dist = self.cosmo.cosmo.comoving_radial_distance(self.zmin)
        max_dist = self.cosmo.cosmo.comoving_radial_distance(self.zmax)
        inds_in = np.where(np.logical_and(dist_edges[:-1] >= min_dist, dist_edges[1:] <= max_dist))[0]
        N_in = len(inds_in)
        #open the first one
        fil = fits.open(fnames[sort_ind[inds_in[0]]])
        #Start the catalog appending everything
        bigcat = np.array(fil[1].data)
        #Open the rest and append
        for ifile in range(1,N_in):
            print(fnames[ifile])
            fil = fits.open(fnames[sort_ind[inds_in[ifile]]])
            bigcat = np.append(bigcat, np.array(fil[1].data))
            
        self.halo_catalog = bigcat
        return
        
    @cached_lightcone_property
    def halo_luminosity(self):
        '''
        Computes the halo luminosity for each of the lines of interest
        '''
        L_line_halo = {}
        #Get the SFR
        if self.do_external_SFR:
            #convert halo mass to Msun
            Mhalo_Msun = self.halo_catalog['M_HALO']*self.Msunh  
            SFR = getattr(extSFRs,self.external_SFR)(Mhalo_Msun.value,self.halo_catalog['Z'])
        else:
            SFR = self.halo_catalog['SFR_HALO']
            
            
        for line in lines.keys():
            if lines[line]:
                L_line_halo[line] = getattr(LM,models[line]['model_name'])(self,self.SFR,self.models[line]['model_pars'])
                
        self.L_line_halo = L_line_halo
        
        return
    
    def make_lightcone(self):
        '''
        Wrapper for "read_halo_catalog" and "halo_luminosity"
        '''
        self.read_halo_catalog()
        self.halo_luminosity()
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
        if any(item in lightcone_params for item in new_params.keys()):
            for attribute in self._update_lightcone_list:
                delattr(self,attribute)
            self._update_lightcone_list = []
        if any(item in survey_params for item in new_params.keys()):
            for attribute in self._update_survey_list:
                delattr(self,attribute)
            self._update_survey_list = []
        #update parameters
        for key in new_params:
            setattr(self, key, new_params[key])
            
        return
    

