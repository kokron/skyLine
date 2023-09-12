# SkyLine

SkyLine is a python code that generates mock line-intensity maps (both in 3D and 2D) in a lightcone from a halo catalog, accounting for the evolution of clustering and astrophysical properties, and observational effects such as spectral and angular resolutions, line-interlopers, galactic foregrounds, etc. Using a given astrophysical model for the luminosity of each line, SkyLine paints the signal for each emitter and generates the map, adding coherently all contributions of interest. In addition, SkyLine can generate maps with the distribution of Luminous Red Galaxies and Emitting Line Galaxies.

SkyLine has been presented in [arXiv:2212.08056](https://arxiv.org/abs/2212.08056) and provides a flexible playground to investigate the potential of LIM techniques and to test and improve methods to maximize the return of LIM surveys. A list including some of the works employing SkyLine can be found [here]().

SkyLine can include any spectral line sourced within collapsed dark matter halos (hence excluding pre-reionization 21cm from the IGM). Thanks to its modular structure, it is very easy to implement spectral lines and astrophysical models relating their luminosity to the astrophysical properties sof the halo.

The default halo catalog used is obtained from the best fit of [UniverseMachine](https://bitbucket.org/pbehroozi/universemachine/src/main/) for the [MultiDark Planck 2](https://www.cosmosim.org/cms/simulations/mdpl2/)  simulations, although any other catalog with similar structure (see below) can be used. The default catalog corresponds to a lightcone generated with the same rotation matrices as the [Agora Project](https://yomori.github.io/agora/index.html). Therefore, one of the main benefits of using the default catalog involves the possibility of cross correlating the SkyLine LIM mocks with CMB secondary anisotropies and radio galaxies.

## Installation

Clone the SkyLine repository using git or download it, and add to your pythonpath (in e.g., bash_rc or bash_profile) the path pointing to the main folder of the repository.

## Prerequisites

SkyLine requires several packages which should be familiar to astronomers, including numpy, scipy, and astropy.  Astropy units are used throughout this code to avoid unit conversion errors. 

SkyLine maps are generated using existing python packages. 3-dimensional maps require [pmesh](https://rainwoodman.github.io/pmesh/) to be generated, and [nbodykit](https://nbodykit.readthedocs.io/en/latest/) to compute the power spectra. In turn, 2-dimensional maps are created using [healpy](https://healpy.readthedocs.io/en/latest/index.html). In addition, SkyLine uses [dask](https://docs.dask.org/en/stable/) to speed up the computations of large arrays.

If foregrounds want to be added and computed as SkyLine runs (they can also be input as files to be read by SkyLine), [PySM3](https://pysm3.readthedocs.io/en/latest/) is required.

Finally, SkyLine uses the python [camb](https://camb.readthedocs.io/en/latest/) wrapper to compute all needed cosmological quantities. 

## Quickstart

After adding the SkyLine folder to your python path, you can quickly generate a map (for instance for the CO(1-0) line):

```
from SkyLine import make_measurements

input_params=dict(halo_lightcone_dir=<your path>, lines=dict(CO_J10=True), target_line = 'CO_J10', 
models=dict(CO_J10=dict(model_name='CO_lines_scaling_LFIR',
            model_pars={'alpha':0.67403184,'beta':4.89800039,'alpha_std':0,'beta_std':0,'sigma_L':0.2})),
            LIR_pars = dict(IRX_name='Bouwens2020', log10Ms_IRX=9.15, alpha_IRX=0.97, sigma_IRX=0.2,
            K_IR=0.63*1.73e-10, K_UV=0.63*2.5e-10),                          
                             do_intensity=False)
LC = make_measurements(input_params)

```

You can check all input parameters and their descriptions in `input_param_description.md`, and all the quantities that can be computed within `SkyLine` are commented in `source/lightcone.py`, `source/survey.py` and `source/measurements.py`.

All modules in SkyLine use an update(), which allows parameter values to be changed after creating the model.  Most outputs are created as @cached_properties, which will update themselves using the new value after update() is called.  For example, to change the maximum frequency observed you could run

```
LC = make_measurements(input_params)
LC.update(nuObs_max=45*u.GHz)

```

The update() method is somewhat optimized, in that it will only rerun functions if required.  WARNING: Note that when running without caching the halo catalog (i.e., if `cache_catalog = False`) you should not used the update() method to avoid issues.

### Examples

An ipython notebook fully commented is provided as an example. 

### Modules

SkyLine main functions read a dict of parameters and creates an object which computes desired quantities and generates the maps from those parameters.  The object created can come from one of several modules. The main modules (included in the `source` folder) are:

- `lightcone.py`: It reads the halo catalog and makes the initial cut to the halo lightcone. It computes the luminosities for each line and halo, as well as the observed frequencies of each line. 
- `survey.py`: It generates the desired mock map in 2D or 3D, according to the survey parameter input, including observational effects. 
- `measurements.py`: It computes the power spectrum from the generated map
- `line_models.py`: It contains the implemented spectral line and the corresponding models for the relation between the line luminosity and the halo properties. Here it is possible to add new lines and models. 
- `external_sfrs.py`: It contains models with star formation rate relations as function of halo mass, or routines to read and interpolate SFR(M,z) tables.
- `utilities.py`: It containts general functions that are used in the other modules.it 

### Line Emission Models

All line-emission models implemented are in `source/line_models.py`. Currently, all models are based on different empirical relations connecting line luminosities with halo mass, star-formation rate and/or infrared luminosity, but more theory-motivated models can be implemented using similar functions. 

### Custom halo catalog

You can input any halo catalog of your choice, always that, to the very least, fulfills the following conditions: 

- Columns names `RA`, `DEC`, `Z`, `DZ`, `M_HALO`, corresponding to right ascencion [deg], declination [deg], cosmological redshift, redshift due to peculiar velocities, halo mass (in Msun/h units). In addition, it can contain columns name `SFR_HALO` and `SM_HALO`, for star-formation rate (in Msun/year units) and stellar mass (in Msun units), respectively. Note that the star formation can be computed using external relations.
- Each file of the catalog must correspond to slices in redshift corresponding to 25 Mpc/h in comoving distance, and have the same name with the exception of the last number, ordered in ascending order as the slice corresponds to higher redshift. 
- If using different cosmological parameters than MDPL2, please make sure to modify them in `lightcone.py`.

### Fiducial halo catalogs

We make public two fiducial halo light cones that were used in [arXiv:2212.08056](https://arxiv.org/abs/2212.08056). They contain UniverseMachine outputs as a function of radial distance from the center of the light cone. SkyLine can automatically read these in sequentially in order to construct 2D and 3D maps of different line intensities (or number counts maps, as a consistency check). These halo catalogs are derived from the light cones of the Agora simulations [arXiv:2212.07420](https://arxiv.org/abs/2212.07420) and thus are fully consistent with the simulation outputs of Agora. 

The first catalog may be found [here](https://app.globus.org/file-manager?origin_id=37653861-6130-4f67-85c2-fd208003820f&origin_path=%2F). It contains a 20 deg x 20 deg selection around RA x DEC, resulting in a 400 deg^2 catalog, among the largest publicly available for LIM studies. Halos can be selected out to $z\sim10$ but we caution that the SFR estimates become increasingly inaccurate at higher redshifts (as pointed out in Appendix A of the skyLine paper). 

The second halo catalog is a full sky realization of the above, found [here](https://app.globus.org/file-manager?origin_id=17dae81e-dae6-43da-ba18-260952692993&origin_path=%2F).

## Usage

When used, please refer to this github page and cite [arXiv:2212.08056](https://arxiv.org/abs/2212.08056)

## Authors

* **Gabriela Sato-Polito**
* **Nickolas Kokron**
* **José Luis Bernal**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
