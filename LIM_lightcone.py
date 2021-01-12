'''
Primary functions to call the modules
'''

from source.lightcone import Lightcone
from source.survey import Survey

try:
    import yaml
except:
    print('Warning, you will only be able to input parameters with a dictionary')


def paint_lightcone(input_pars=None):
    '''
    Base function to initiate Lightcone objects. 
    
    -input_pars:    String containing the path to a input file (*.yaml) or a
                    dictionary containing the input parameters. 
    '''

    #check input parameters given, and read them
    if type(input_pars) == str:
        with open(input_pars) as f:
            pars = yaml.load(f, Loader=yaml.FullLoader)
        if 'output_root' not in pars:
            pars['output_root'] = 'output/'+input_pars.split('/')[-1].split('.')[0]
    elif type(input_pars) == dict:
        pars = input_pars
    else:
        raise ValueError('Please input a dictionary or a path to a *.yaml file with the input parameters.')
    
    #Return the run object
    return Lightcone(**pars)
    
def make_survey(input_pars=None):
    '''
    Base function to initiate the Survey object (which embeds Lightcone)
    
    -input_pars:    String containing the path to a input file (*.yaml) or a
                    dictionary containing the input parameters. Must include also
                    the parameters for paint_lightcone
    '''
    #check input parameters given, and read them
    if type(input_pars) == str:
        with open(input_pars) as f:
            pars = yaml.load(f, Loader=yaml.FullLoader)
        if 'output_root' not in pars:
            pars['output_root'] = 'output/'+input_pars.split('/')[-1].split('.')[0]
    elif type(input_pars) == dict:
        pars = input_pars
    else:
        raise ValueError('Please input a dictionary or a path to a *.yaml file with the input parameters.')
    
    #Return the run object
    return Survey(**pars)
    



