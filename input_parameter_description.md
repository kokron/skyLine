# INPUT PARAMETERS

Here we briefly describe all the input parameters that can be set for SkyLine, specifying the default parameters.

### Generation of lightcone to paint

- **halo_lightcone_dir**: Path to the directory containing all files related to the halo lightcone catalog

- **lightcone_slice_width**: Width (in Mpc/h) of each of the lightcone slice files in halo_lightcone_dir. Input the value without astropy units (conversion from Mpc/h to Mpc done internally). (Default: 25)

- **zmin,zmax**: Minimum and maximum redshifts to read from the lightcone (default: 0,20 - limited by Universe Machine)

- **RA_width**: Total RA width to read from the lightcone. Assumed to be centered in origin (Default = 2 deg)

- **DEC_width**: Total DEC to read from the lightcone. Assumed to be centered in origin (Default = 2 deg)

- **lines**: What lines are painted in the lightcone. Dictionary with bool values (default: All false). Check available lines in source/line_models.py

- **models**: Models for each line. Dictionary of dictionaries (first layer, same components of "lines", second layer, the following components: model_name, model_pars (depends on the model)) (default: empty dictionary). Check available models in source/line_models.py
                        
- **LIR_pars**: Dictionary with the parameters required to compute infrared luminosity, needed to compute certain lines luminosities. Check the LIR function in source/line_models.py for the required parameters and available models

- **do_external_SFR**: Boolean, whether to use a SFR different than Universe Machine (default:False)

- **external_SFR**: SFR table to interpolate or fitting function

- **sig_extSFR**: log-scatter for an external SFR

- **seed**: seed for the RNG object

- **cache_catalog**: Boolean, whether to read all halo files at one and keep the whole catalog in cache or read iteratively each time. (default: True). Useful when the footprint and redshift range is small, for large sky areas, this **must** be False for memory usage reasons. Can also be a good idea for interlopers to reduce memory usage, but losing the cache functionality


### Survey and sample parameters

- **do_intensity**: Bool, if True quantities are output in specific temperature (Jy/sr units) rather than brightness temperature (muK units) (Default = False)

- **Tsys**: Instrument sensitivity. System temperature for brightness temperature and noise equivalent intensitiy (NEI) for intensitiy (Default = 40 K)

- **Nfeeds**: Total number of feeds (detector*antennas*polarizations) (Default = 19)

- **nuObs_min,nuObs_max**: minimum and maximum ends of the frequency band (Default = 26-34 GHz)

- **RAObs_width**: Total RA width observed. Assumed centered at 0 (Default = 2 deg)

- **DECObs_width**: Total DEC width observed. Assumed centered at 0 (Default = 2 deg)

- **dnu**: Width of a single frequency channel (Default = 15.6 MHz)

- **beam_FWHM**: Beam full width at half maximum (Default = 4.1 arcmin)

- **tobs**: Observing time on a single field (Default = 6000 hr)

- **target_line**: Target line of the survey (Default: CO)

- **v_of_M**: Function returning the unitful FWHM of the line profile of emission given halo mass. Line widths are not applied if v_of_M is None. Only relevant if do_angular = False and number_count = False. (default = None) (example: `lambda M:50*u.km/u.s*(M/1e10/u.Msun)**(1/3.)`)
                    
- **line_incli**: Bool, if accounting for randomly inclined line profiles. (default = True; does not matter if v_of_M = None or do_angular = True or number_count = True)

- **Nsigma_v_of_M**: Number of bins in sigma_v_of_M for a coarse smoothing to model the line broadening (default = 10; only relevalant if v_of_M != None and do_angular = False and number_count = False)

- **angular_supersample**: Factor of supersample with respect to the survey angular resolution when making the grid. Important to have good power spectrum multipole calculations (e.g., enough mu values as function of kmax). (Default: 5)
                        
- **spectral_supersample**: Factor of supersample with respect to the survey spectral resolution when making the grid. Important to have good power spectrum multipole calculations (e.g., enough mu values as function of kmax). (Default: 1)

- **do_angular_smooth**: Boolean: apply smoothing filter to implement angular resolution limitations. (Default: True)

- **do_spectral_smooth**: Boolean: apply smoothing filter to implement spectral resolution limitations. (Default: False)

- **kind_spectral_smooth**: Whether the spectral smoothing is done using a top-hat filter or a Gaussian filter ('tophat' or 'gaussian', respectively. Default: 'tophat')
                        
- **cube_mode**: Mode to create the data rectangular cube for 3d maps (irrelevant if do_angular == True). Options are:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 'outer_cube': The lightcone is inscribed in the cube
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 'inner_cube': The cube is inscribed in the lightcone
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 'mid_redshift': coordinates transverse to line of sight are obtained using a single redshift for all emitters (for each line)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 'flat_sky': Applies flat sky approximation on top of 'mid_redshift'
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Default: 'inner_cube')
                        
- **do_z_buffering**: Boolean: add higher z emitters to fill corners at high-z end. Only relevant if cube_mode = 'inner_cube' or 'mid_redshift'. (Default: True)

- **do_downsample**: Boolean: Downsample the map such as supersample=1. (Default: True; make if False for nice plots)

- **do_remove_mean**: Boolean: Remove the mean of the map or not (Defult: True)

- **do_angular**: Create an angular survey (healpy map) (Default: False)

- **nside**: NSIDE used by healpy to create angular maps. (Default: 2048)

- **mode**: String: what kind of map you want to simulate. Options: 'lim' and 'number_count' (for galaxy density). Default: 'lim'. If mode = number_count, it allows to use all galaxies or select between lrgs, elgs, and all

- **Mhalo_min**: Minimum halo mass (in Msun/h) to be included in the survey (filter for halos_in_survey). Default:0

- **Mstar_min**: Minimum stellar mass in a halo (in Msun) to be ncluded in the survey (filter for halos_in_survey). Default:0

- **gal_type**: Whether to select only LRGs or ELGs, or all galaxies. Options: Options: 'all', 'lrg', 'elg'. Irrelevant if mode != number_count

- **dngaldz_file**: File containing a table with the redshift distribution of galaxy number density if number_count = True. Irrelevant otherwise. Input a file in table to interpolate and normalize. Format: 2 columns with z, dNdz    (Default: None -> must have one! Will be expected to be in Mpc**-3 or sr**-1 if angular map)   

- **resampler**: Set the resampling window for the 3d maps (Irrelevant if do_angular=True). (Default: 'cic')


### Measurement parameters

- **dk**: k spacing for the power spectrum (default: None -> to be set by nbodykit)

- **kmin,kmax**: Minimum and maximum k values for the power spectrum (default: 0., None -> to be set by nbodykit)

- **Nmu**: Number of sampling in mu to compute the power spectrum (default: 10)
                        
- **lmax**: Maximum multipole to compute the angular power spectrum. Default=1000

- **remove_noise**: Remove the expected instrumental noise power spectrum (sigma_N^2*Vvox) from the observed power spectrum. (default: False)
                        
- **angular_map**: Whether the map used is angular (healpy map). (Default: False)

- **do_read_map**: Whether to read a map already saved from a survey computed before (Default:False)

- **map_name**: The name of the map to read (Default: '')
