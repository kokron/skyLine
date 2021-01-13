'''
Base module to pain the LIM lightcone
'''

import numpy as np
from glob import glob
from astropy.io import fits

import source.line_models as LM
import source.external_sfrs as extSFRs

from source.utilities import check_params,check_models,check_sfr

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
                 lines = dict(CO = False, CII = False, Halpha = False, Lyalpha = False, HI = False),
                 models = dict(CO = dict(model_name = '', model_pars = {}), CII = dict(model_name = '', model_pars = {}),
                               Halpha = dict(model_name = '', model_pars = {}), Lyalpha = dict(model_name = '', model_pars = {}), 
                               HI = dict(model_name = '', model_pars = {})),
                 do_external_SFR = False, external_SFR = '',
                 output_root = "output/default"):
                 
        # Get list of input values to check type and units
        self._run_params = locals()
        self._run_params.pop('self')
        
        # Get list of input names and default values
        self._default_run_params = get_default_params(Run.__init__)
        # Check that input values have the correct type and units
        check_params(self._run_params,self._default_run_params)
        # Fill lines no included with false
        for key in list(self._default_run_params['lines'].keys()):
            if key not in self._run_params['lines'].keys():
                self._run_params['lines'][key] = False
        
        # Set all given parameters
        for key in self._run_params:
            setattr(self,key,self._run_params[key])
                                
        # Check that the input line models are included
        check_models(self.lines,self.models)
        if self.do_external_SFR:
            check_sfr(self.external_SFR)
            
        #Placeholder
        self.L_line_halo = None
        
        
    def read_halo_catalog(self):
        '''
        Reads all the files from the halo catalog and appends the slices
        '''
        fnames = glob(self.halo_lightcone_dir+'/*')
        #open the first one
        fil = fits.open(fnames[0])
        #Start the catalog appending everything
        bigcat = np.array(fil[1].data)
        #Open the rest and append
        for ifile in range(1,len(fnames)):
            fil = fits.open(fnames[i])
            bigcat = np.append(bigcat, np.array(fil[1].data))
            
        self.halo_catalog = bigcat
        return
        
        
    def halo_luminosity(self):
        '''
        Computes the halo luminosity for each of the lines of interest
        '''
        L_line_halo = {}
        #Get the SFR
        if self.do_external_SFR:
            ####CHECK UNITS OF MASS!!!!#####
            SFR = getattr(extSFRs,self.external_SFR)(self.halo_catalog['M_HALO'],self.halo_catalog['Z'])
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
        
                
                
    

