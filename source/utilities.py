'''
Set of functions useful in some modules
'''

import source.line_models as LM
import source.external_sfrs as extSFRs
import inspect
import astropy.units as u
import numpy as np
import healpy as hp
from warnings import warn

try:
    import pysm3
    NoPySM = False
except:
    NoPySM = True

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

def check_params(self,input_params, default_params):
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
            elif key=='v_of_M':
                if callable(input_value):
                    pass
            elif key == 'kmax' or key == 'dk':
                if type(input_value) == u.quantity.Quantity:
                    continue
            elif key == 'dNgaldz_file':
                if type(input_value) == str:
                    continue
            elif key == 'spectral_transmission_file':
                if type(input_value) == str:
                    continue
            elif key == 'dnu':
                if type(input_value) == float:
                    continue
            elif key == 'flux_detection_lim':
                if type(input_value) == u.quantity.Quantity or type(input_value) == function:
                    continue
            elif key == 'nu_c':
                if type(input_value) == u.quantity.Quantity:
                    continue
            else:
                raise TypeError("Parameter "+key+" must be a "+
                                str(type(default_value)))

        # Special requirements for some parameters
        line_dict = getattr(LM,'lines_included')(self)
        if key == 'lines':
            for line in input_value.keys():
                if line not in line_dict:
                    raise ValueError('The line {} is currently not included in the code. Please correct or modify "lines_included" in line_models.py'.format(line))
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
            if input_value not in line_dict:
                    raise ValueError('The line {} is currently not included in the code. Please correct or modify "lines_included" in line_models.py'.format(input_value))
                    
        elif key == 'kind_spectral_smooth':
            options = ['tophat','gaussian']
            if input_value not in options:
                raise ValueError('The kind_spectral_smooth input {} is not implemented. Please choose among {} or implement the filter in survey.py'.format(input_value,options))
                
    return
    
def check_updated_params(self):
    '''
    Set of checks for consistency between parameters after update
    '''
    #check unit convention
    unit_conventions = ['Tb','Tcmb','Inu']
    if self.unit_convention not in unit_conventions:
        raise ValueError('The unit convention must be one of {}'.format(unit_conventions))
    
    # Check that the observed footprint is contained in the lightcone
    if self.RAObs_min < self.RA_min or self.RAObs_max > self.RA_max or \
        self.DECObs_min < self.DEC_min or self.DECObs_max > self.DEC_max:
        warn('Please, your observed limits RA_Obs=[{},{}], DEC_Obs=[{},{}] must be within the lightcone limits RA=[{},{}], DEC=[{},{}].'.format(self.RAObs_min,self.RAObs_max,self.DECObs_min,self.DECObs_max,self.RA_min,self.RA_max,self.DEC_min,self.DEC_max))

    # Check that the bandwidth and lines used are included in the lightcone limits
    if self.mode == 'lim':
        for line in self.lines.keys():
            if self.lines[line]:
                zlims = (self.line_nu0[line].value)/np.array([self.nuObs_max.value,self.nuObs_min.value])-1
                if zlims[0] <= self.zmin or zlims [1] >= self.zmax:
                    warn('The line {} on the bandwidth [{:.2f},{:.2f}] corresponds to z range [{:.2f},{:.2f}], while the included redshifts in the lightcone are within [{:.2f},{:.2f}]. Please remove the line, increase the zmin,zmax range or reduce the bandwith.'.format(line,self.nuObs_max,self.nuObs_min,zlims[0],zlims[1],self.zmin,self.zmax))

    #Check healpy pixel size just in case:
    if self.do_angular:
        npix_fullsky = 4*np.pi/((self.beam_FWHM/self.angular_supersample)**2).to(u.sr).value
        min_nside = hp.pixelfunc.get_min_valid_nside(npix_fullsky)
        if (min_nside > self.nside):
            warn("The minimum NSIDE to account for beam_FWHM*angular_supersample is {}, but NSIDE={} was input.".format(min_nside,self.nside))

    self.cube_mode_options = ['outer_cube','inner_cube','mid_redshift','flat_sky']
    if self.cube_mode not in self.cube_mode_options:
        raise ValueError('The cube_mode choice must be one of {}'.format(self.cube_mode_options))
        
    if self.mode not in ['lim','number_count','cib']:
        raise ValueError('mode input must be one of {}'.format(['lim','number_count','cib']))
    
    if self.mode == 'number_count':
        if self.gal_type not in ['all','lrg','elg']:
            raise ValueError('gal_type input must be one of {}'.format(['all','lrg','elg']))
        if self.dNgaldz_file == None:
            raise ValueError('Please input a file with the number density per redshift')
        if type(self.dnu) == u.quantity.Quantity:
            raise ValueError('If mode == number_count, dnu must be dimensionless (indicating the width in redshfit of the 3d cell)')

    if self.mode == 'cib' or ((self.do_angular and self.unit_convention == 'Tcmb') and self.mode != 'number_count'):
        if self.spectral_transmission_file == None:
            raise ValueError('Please input a file with the spectral transmission')
    
    if NoPySM and self.do_gal_foregrounds==True:
        raise ValueError('PySM must be installed to model galactic foregrounds')

    #Set units for observable depending on convention
    if self.unit_convention == 'Inu':
        self.unit = u.Jy/u.sr
    else:
        self.unit = u.uK

    if self.mode == 'number_count':
        self.unit = None

    if self.do_angular != self.angular_map:
        raise ValueError("'do_angular' and 'angular_map' must be the same when the map is computed")
        
    if NoPySM and self.do_gal_foregrounds==True:
        raise ValueError('PySM must be installed to model galactic foregrounds')
        
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

def CompensateNGPShotnoise(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
    the gridded field with the NGP window function in configuration space,
    as well as the approximate aliasing correction to the first order

    For NGP this is just 1. 

    .. note::
        see equation 20 of
        `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

    Parameters
    ----------
    w : list of arrays
        the list of "circular" coordinate arrays, ranging from
        :math:`[-\pi, \pi)`.
    v : array_like
        the field array
    """
    return v

def newton_root(fun,fun_prime,x0,*args,Niter=5):
    '''
    Quick implementation of the Newton-Raphson method for arrays
    Usage: newton_root(<your function>,<derivative of the function>,
                        <initial point>,<arg1>,...,<argN>,Niter=<Niter>)
    fun and fun_prime must be defined with the same number and order of arguments
    arg1 starts counting after the variable to find the root for (which should be the first)
    '''
    for i in range(Niter):
        x0 = x0 - fun(x0,*args)/fun_prime(x0,*args)
    return x0