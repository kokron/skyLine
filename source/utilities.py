'''
Set of functions useful in some modules
'''

import source.line_models as LM
import source.external_sfrs as extSFRs
import inspect
from lim import lim
import astropy.units as u
import numpy as np
import healpy as hp

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
        
class cached_read_property(object):
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
        instance._update_read_list.append(self.func.__name__)

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
        
class cached_measure_property(object):
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
        instance._update_measure_list.append(self.func.__name__)

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
            if key == 'seed':
                if type(input_value) == int:
                    continue

            else:
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
            lines_available = ['CO_J10','CII','Halpha','Hbeta','Lyalpha','HI',
                              'CO_J21','CO_J32','CO_J43','CO_J54','CO_J65','CO_J76',
                              'NIII','NII','OIII_88','OI_63','OI_145','OII','OIII_0p5']
            if input_value not in lines_available:
                raise ValueError('The target line {} must be one of the available lines: {}'.format(input_value,lines_available))

    return
    
def check_updated_params(self):
    '''
    Set of checks for consistency between parameters after update
    '''
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
        npix_fullsky = 4*np.pi/((self.beam_FWHM/self.angular_supersample)**2).to(u.sr).value
        min_nside = hp.pixelfunc.get_min_valid_nside(npix_fullsky)
        if (min_nside > self.nside):
            print("WARNING!!! the minimum NSIDE to account for beam_FWHM*angular_supersample is {}, but NSIDE={} was input.".format(min_nside,self.nside))
        #Avoid inner cut if do_angular:
        if self.do_angular and self.do_inner_cut:
            raise ValueError('If you want to work with angular maps, you do not need the inner cut, hence please use do_inner_cut = False')

    #Set units for observable depending on convention
    if self.do_intensity:
        self.unit = u.Jy/u.sr
    else:
        self.unit = u.uK
        
    if self.do_angular != self.angular_map:
        raise ValueError("'do_angular' and 'angular_map' must be the same when the map is computed")
        
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


def merge_dicts(D):
    '''
    Merges dictionaries
    '''
    dic = {}
    for k in D:
        dic.update(k)
    return dic
