'''
Set of functions useful in some modules
'''

import source.line_models as LM
import source.external_sfrs as extSFRs
import inspect
from lim import lim
import astropy.units as u
import numpy as np

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
        if (self.beam_FWHM.to(u.arcmin)).value < hp.nside2resol(self.nside, arcmin=True):
            print("WARNING!!! the healpy pixel side chosen, from NSIDE = {}, is {:.2f} times bigger than the beam_FWHM. Consider increasing NSIDE (remember that it must be a power of 2)".format(self.nside,hp.nside2resol(self.nside, arcmin=True)/self.beam_FWHM.to(u.arcmin)).value)
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


def dict_lines(self,name,pars):
    '''
    Translates between the conventions of this code and lim, and returns
    the model name and model parameters for each case
    '''
    if name == 'CO_Li16':
        model_name = 'TonyLi'
        model_pars = dict(alpha=pars['alpha'],beta=pars['beta'],
                          dMF=pars['delta_mf'],sig_SFR=self.sig_extSFR)
        if self.do_external_SFR:
            model_pars['SFR_file'] = '../SFR_tables/sfr_release.dat'
        else:
            model_pars['SFR_file'] = '../SFR_tables/UM_sfr.dat'
    elif name == 'CII_Silva15':
        model_name = 'SilvaCII'
        model_pars = dict(a=pars['aLCII'],beta=pars['bLCII'])
        if self.do_external_SFR:
            model_pars['SFR_file'] = '../SFR_tables/Silva15_SFR_params.dat'
        else:
            model_pars['SFR_file'] = '../SFR_tables/UM_sfr.dat'
    elif name == 'Halpha_Gong17':
        model_name = 'GongHalpha'
        model_pars = dict(K_Halpha=pars['K_Halpha'],Aext=pars['Aext_Halpha'])
        if self.do_external_SFR:
            model_pars['SFR_file'] = '../SFR_tables/Gong16_SFR_params.dat'
        else:
            model_pars['SFR_file'] = '../SFR_tables/UM_sfr.dat'
    elif name == 'Hbeta_Gong17':
        model_name = 'GongHbeta'
        model_pars = dict(K_Hbeta=pars['K_Hbeta'],Aext=pars['Aext_Hbeta'])
        if self.do_external_SFR:
            model_pars['SFR_file'] = '../SFR_tables/Gong16_SFR_params.dat'
        else:
            model_pars['SFR_file'] = '../SFR_tables/UM_sfr.dat'
    elif name == 'OIII_Gong17':
        model_name = 'GongOIII'
        model_pars = dict(K_OIII=pars['K_OIII'],Aext=pars['Aext_OIII'])
        if self.do_external_SFR:
            model_pars['SFR_file'] = '../SFR_tables/Gong16_SFR_params.dat'
        else:
            model_pars['SFR_file'] = '../SFR_tables/UM_sfr.dat'
    elif name == 'OII_Gong17':
        model_name = 'GongOII'
        model_pars = dict(K_OII=pars['K_OII'],Aext=pars['Aext_OII'])
        if self.do_external_SFR:
            model_pars['SFR_file'] = '../SFR_tables/Gong16_SFR_params.dat'
        else:
            model_pars['SFR_file'] = '../SFR_tables/UM_sfr.dat'
    else:
        raise ValueError('The input astrophysical model has no equivalent in the lim code.')


    return model_name, model_pars


def set_lim(self):
    '''
    Calls to lim to compute theoretical lim quantities
    '''
    fid = dict(cosmo_input_camb={'H0':67.8,'ombh2':0.02312,'omch2':0.118002988,
                      'As':2.23832e-9,'ns':0.96,'mnu':0.06})
    krange = {'nk':512,'kmin':1e-5*u.Mpc**-1,'kmax':10*u.Mpc**-1,'k_kind':'log','nmu':10000,'smooth':self.do_smooth}
    hmf = dict(model_type='ML',hmf_model='Tinker',bias_model='Tinker10',do_onehalo=True)

    astromodel=dict(nu=self.line_nu0[self.target_line],model_name="TonyLi",model_par={'alpha': 1.37, 'beta': -1.74, 'dMF': 1, 'sig_SFR': 0.3, 'SFR_file': '../SFR_tables/sfr_table_UniverseMachine_clean.dat'}, sigma_scatter = 0.3)
    survey = dict(Tsys_NEFD=self.Tsys,do_Jysr=self.do_intensity,tobs=self.tobs,Omega_field=self.Omega_field,
                  nuObs=self.nuObs_mean,Delta_nu=self.delta_nuObs,dnu=self.dnu,beam_FWHM=self.beam_FWHM,
                  Nfeeds=self.Nfeeds,Nfield=1)
    M = lim(merge_dicts([fid,krange,hmf,astromodel,survey]))

    return M
