'''
Set of functions useful in some modules
'''

import source.line_models as LM
import source.external_sfrs as extSFRs
import inspect

class cached_lightcone_property(object):
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

        # ADDED THIS CODE TO LIST PROPERTY FOR UPDATING
        instance._update_lightcone_list.append(self.func.__name__)
        
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res
        
class cached_survey_property(object):
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

        # ADDED THIS CODE TO LIST PROPERTY FOR UPDATING
        instance._update_survey_list.append(self.func.__name__)
        
        res = instance.__dict__[self.func.__name__] = self.func(instance)
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
                if input_value[line]:
                    if input_params['models'][line]['model_name'] == '':
                        raise ValueError('Please input a "model_name" within "models" for the {} line.'.format(line))
                    elif not hasattr(LM,input_params['models'][line]['model_name']):
                        raise ValueError('{} not found in line_models.py'.format(input_params['models'][line]['model_name']))
                    if input_params['models'][line]['model_pars'] == {}:
                        raise ValueError('Please input the parameters of the model in "model_pars" within "models" for the {} line.'.format(line))
        elif key == 'do_external_SFR':
            if input_value and not hasattr(extSFRs,input_params['external_SFR']):
                raise ValueError('{} not found in external_sfrs.py'.format(input_params['external_SFR']))
                
        elif key == 'target_line':
            lines_available = ['CO','CII','Halpha','Lyalpha','HI']
            if input_value not in lines_available:
                raise ValueError('The target line {} must be one of the available lines: {}'.format(input_value,lines_available))
                
    return
        

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
    
def aniso_filter(k, v):
    '''
    Filter for k_perp and k_par modes separately. 
    Applies to an nbodykit mesh object as a regular filter.
    
    Uses globally defined variables:
        Rperp - 'angular' smoothing in the flat sky approximation
        Rpar - 'radial' smoothing from number of channels.
    
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
