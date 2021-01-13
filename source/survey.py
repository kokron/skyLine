'''
Base module to make a LIM survey from painted lightcone
'''

import numpy as np

from source.lightcone import Lightcone

class Survey(object):
    '''
    An object controlling all relevant quantities needed to create the 
    painted LIM lightcone. It reads a lightcone catalog of halos with SFR 
    quantities and paint it with as many lines as desired.
    
    Allows to compute summary statistics as power spectrum and the VID for 
    the signal (i.e., without including observational effects).
        
    INPUT PARAMETERS:
    ------------------
    
    -output_root            Root path for output products. (default: output/default)                                
    '''
    def __init__(self,
                 output_root = "output/default"):

        return
                 
        
