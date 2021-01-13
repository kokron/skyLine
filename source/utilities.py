'''
Set of functions useful in some modules
'''

import source.line_models as LM
import source.external_sfrs as extSFRs

def check_params(input_params, default_params):
    '''
    Check input parameter values to ensure that they have the required type
    '''
    
    for key in input_params.keys():
        # Check if input is a valid parameter
        if key not in default_params.keys():
            raise AttributeError(key+" is not a valid parameter")
        
        input_value = input_params[key]
        default_value = default_params[key]
        
        # Check if input has the correct type
        if type(input_value)!=type(default_value):
            raise TypeError("Parameter "+key+" must be a "+
                                str(type(default_value)))
                                    
        # Special requirements for some parameters
        if key == 'lines':
            for line in input_value.keys():
                if line:
                    if input_params[line+'_model'] == {}:
                        raise ValueError('Please enter input parameters for the {} model using the "{}_model" input dictionary'.format(line,line))




def check_models(lines,models):
    '''
    Check that incompatible likelihoods are not included at the same time
    '''
    for line in lines.keys():
        if lines[line] and not hasattr(LM,models[line]['model_name']):
            raise ValueError('{} not found in line_models.py'.format(models[line]['model_name']))
        

def check_sfr(sfr):
    '''
    Check that incompatible likelihoods are not included at the same time
    '''
    if sfr and not hasattr(extSFRs,sfr):
        raise ValueError('{} not found in external_sfrs.py'.format(sfr))

