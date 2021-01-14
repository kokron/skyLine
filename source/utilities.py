'''
Set of functions useful in some modules
'''

import source.line_models as LM
import source.external_sfrs as extSFRs
import inspect

class cached_property(object):
    """
    From github.com/Django, who wrote a much better version of this than
    the one I had previously.
    
    Decorator that converts a self.func with a single self argument into a
    property cached on the instance.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type=None):
        if instance is None:
            return self

        return res

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
                    if input_params['models'][line]['model_name'] == '':
                        raise ValueError('Please input a "model_name" within "models" for the {} line.'.format(line))
                    elif input_params['models'][line]['model_pars'] == {}:
                        raise ValueError('Please input the parameters of the model in "model_pars" within "models" for the {} line.'.format(line))

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


def get_default_params(func):
    '''
    Gets the default parameters of a function or class. Output
    is a dictionary of parameter names and values, removing any 
    potential instance of "self"
    '''
    
    args = inspect.getargspec(func)
    
    param_names = args.args
    if 'self' in param_names:
        param_names.remove('self')
    
    default_values = args.defaults
    
    default_params = dict(zip(param_names,default_values))

    return default_params
