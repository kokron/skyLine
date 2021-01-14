'''
Base module to make a LIM survey from painted lightcone
'''

import numpy as np
import astropy.units as u
import astropy.constants as cu

from source.lightcone import Lightcone
from source.utilities import cached_survey_property,get_default_params,check_params

class Survey(object):
    '''
    An object controlling all relevant quantities needed to create the 
    painted LIM lightcone. It reads a lightcone catalog of halos with SFR 
    quantities and paint it with as many lines as desired.
    
    Allows to compute summary statistics as power spectrum and the VID for 
    the signal (i.e., without including observational effects).
        
    INPUT PARAMETERS:
    ------------------
    
    -do_Intensity           Bool, if True quantities are output in specific temperature
                            (Jy/sr units) rather than brightness temperature 
                            (muK units) 
                            (Default = False)
                    
    -Tsys:                  Instrument system temperature (Default = 40 K)
    
    -Nfeeds:                Number of feeds (Default = 19)
    
    -beam_FWHM:             Beam full width at half maximum (Default = 4.1")
    
    -nuObs_min,nuObs_max:   Total frequency range covered by instrument (Default = 8 GHz)
    
    -dnu:                   Width of a single frequency channel (Default = 15.6 MHz)
    
    -tobs:                  Observing time on a single field (Default = 6000 hr)
    
    -Omega_field:           Solid angle covered by a single field
                            (Default = 2.25 deg^2)    
                            
    -target_line:           Target line of the survey (Default: CO)
    
    -output_root            Root path for output products. (default: output/default)                                
    '''
    def __init__(self,
                 do_Intensity=False,
                 Tsys_NEFD=40*u.K,
                 Nfeeds=19,
                 beam_FWHM=4.1*u.arcmin,
                 nuObs_min = 26*u.GHz,
                 nuObs_max = 34*u.GHz,
                 dnu=15.6*u.MHz,
                 tobs=6000*u.hr, 
                 Omega_field=2.25*u.deg**2,
                 target_line = 'CO',
                 output_root = "output/default",
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
        

        
    @cached_survey_property
    def nuObs_mean(self):
        '''
        Mean observed frequency
        '''
        return 0.5*(self.nuObs_min+self.nuObs_max)
                 
    @cached_survey_property
    def delta_nuObs(self):
        '''
        Experimental frequency bandwith
        '''
        return self.nuObs_max - self.nuObs_min
        
    @cached_survey_property
    def observed_catalog(self):
        '''
        Filters the halo catalog and only takes those that have observed
        frequencies within the experimental frequency bandwitdh
        '''
        #Start for the target line
        inds = np.where(np.logical_and(nuObs_line_halo[self.target_line] >= self.nuObs_min,
                                       nuObs_line_halo[self.target_line] <= self.nuObs_max))[0]
        return 
        
        
