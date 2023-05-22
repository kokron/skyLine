#import sys
#sys.path.append('/home/jlbernal/preLIMinary/')
#sys.path.append('/home/jlbernal/lim')
import numpy as np
import astropy.units as u
import astropy.constants as cu
from SkyLine import make_lightcone, make_survey, make_measurements
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
import dask.array as da


from scipy import integrate
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

LC_path='/home/jlbernal/LightCone_S82_UM/'


def LC_params(line):
    h=0.678
    zmin = 2.5;zmax = 3.5
    zmid=(zmax+zmin)/2
    R=700
    ang_lim=20*u.deg
    model_params=dict(halo_lightcone_dir=LC_path, 
                      do_external_SFR = False, external_SFR = '',
                      SFR_pars=dict(),
                      resampler='cic',
                      angular_supersample=1,
                      spectral_supersample=1,
                      do_downsample=False,
                      cube_mode='inner_cube',
                      Nmu=20,
                      kmax=3*u.Mpc**-1,
                      dk=0.02*u.Mpc**-1,
                      seed=0)
    obs_params=dict(zmin = zmin-0.03, zmax = zmax+0.13, 
                    RAObs_width=ang_lim, DECObs_width=ang_lim,
                    RA_width=ang_lim, DEC_width=ang_lim,
                    Nfeeds=1000, beam_FWHM=2*u.arcmin, tobs=6000*u.hr, 
                    do_angular_smooth=True, do_spectral_smooth=True)
    
    if line=='CO':
        nu_CO=115.271*u.GHz
        dnu=nu_CO/(1+zmid)/R
        model_params.update(dict(lines=dict(CO_J10=True), target_line = 'CO_J10',
                                #models=dict(CO_J10=dict(model_name='CO_Li16',
                                #                    model_pars={'alpha':1.37,'beta':-1.74,'delta_mf':1,'sigma_L':0.3})),
                                 models=dict(CO_J10=dict(model_name='CO_lines_scaling_LFIR',
                                                          model_pars={'alpha':0.81568736,'beta':3.43531917,'alpha_std':0,'beta_std':0,'sigma_L':0.2})),
                                 LIR_pars = dict(IRX_name='Bouwens2020', log10Ms_IRX=9.15, alpha_IRX=0.97, sigma_IRX=0.2,
                                                 K_IR=0.63*1.73e-10, K_UV=0.63*2.5e-10),                          
                                 do_intensity=False))
        obs_params.update(dict(nuObs_max=nu_CO/(1+zmin), nuObs_min=nu_CO/(1+zmax),
                           dnu=dnu, 
                           Tsys=0*u.Jy/u.sr))
        obs_params.update(model_params)
    return obs_params


def halos_in_Tbin(LC, Tbin_min, Tbin_max):
    LC_map=LC.obs_3d_map
    corners=LC_map.x
    
    inds_RA = (LC.halo_catalog_all['RA'] > LC.RAObs_min.value)&(LC.halo_catalog_all['RA'] < LC.RAObs_max.value)
    inds_DEC = (LC.halo_catalog_all['DEC'] > LC.DECObs_min.value)&(LC.halo_catalog_all['DEC'] < LC.DECObs_max.value)
    inds_sky = inds_RA&inds_DEC

    cornerside = (LC.raside_lim[1]**2+LC.decside_lim[1]**2)**0.5
    ang = np.arctan(cornerside/LC.rside_obs_lim[1])
    rbuffer = cornerside/np.sin(ang)
    zbuffer = LC.cosmo.redshift_at_comoving_radial_distance((rbuffer*LC.Mpch).value)
    nu_min = LC.line_nu0['CO_J10']/(zbuffer+1)

    inds = (LC.nuObs_line_halo_all['CO_J10'] >= nu_min)&(LC.nuObs_line_halo_all['CO_J10'] <= LC.nuObs_max)&inds_sky
    halos_in_survey_all = LC.halo_catalog_all[inds]
    
    zmid = (LC.line_nu0[LC.target_line]/LC.nuObs_mean).decompose().value-1
    sigma_par_target = (cu.c*LC.dnu*(1+zmid)/(LC.cosmo.hubble_parameter(zmid)*(u.km/u.Mpc/u.s)*LC.nuObs_mean)).to(LC.Mpch).value

    Lbox = LC.Lbox.value

    Nmesh = np.array([LC.spectral_supersample*np.ceil(Lbox[0]/sigma_par_target),
              LC.angular_supersample*LC.Npixside[0],
              LC.angular_supersample*LC.Npixside[1]], dtype=int)

    ramid = 0.5*(LC.RAObs_max + LC.RAObs_min)
    decmid = 0.5*(LC.DECObs_max + LC.DECObs_min)

    ralim = np.deg2rad(np.array([LC.RAObs_min.value,LC.RAObs_max.value]) - ramid.value) 
    declim = np.deg2rad(np.array([LC.DECObs_min.value,LC.DECObs_max.value]) - decmid.value)
    raside_lim = LC.raside_lim
    decside_lim = LC.decside_lim
    rside_obs_lim = LC.rside_obs_lim

    mins_obs = np.array([rside_obs_lim[0],raside_lim[0],decside_lim[0]])+0.49999*Lbox/Nmesh

    ra,dec,redshift = da.broadcast_arrays(LC.halos_in_survey_all['CO_J10']['RA'], LC.halos_in_survey_all['CO_J10']['DEC'],
                                          LC.halos_in_survey_all['CO_J10']['Zobs'])
    r = redshift.map_blocks(lambda zz: (((LC.cosmo.comoving_radial_distance(zz)*u.Mpc).to(LC.Mpch)).value),
                            dtype=redshift.dtype)
    ra -= ramid.value
    dec -= decmid.value
    ra,dec  = da.deg2rad(ra),da.deg2rad(dec)

    x = da.cos(dec) * da.cos(ra)
    y = da.cos(dec) * da.sin(ra)
    z = da.sin(dec)
    pos = da.vstack([x,y,z]).T                    
    cartesian_halopos = r[:,None] * pos
    lategrid = np.array(cartesian_halopos.compute())

    filtering = (lategrid[:,0] >= rside_obs_lim[0]) & (lategrid[:,0] <= rside_obs_lim[1]) & \
                (lategrid[:,1] >= raside_lim[0]) & (lategrid[:,1] <= raside_lim[1]) & \
                (lategrid[:,2] >= decside_lim[0]) & (lategrid[:,2] <= decside_lim[1])
    lategrid = lategrid[filtering]
    for n in range(3):
        lategrid[:,n] -= mins_obs[n]
        
    Xcorner=np.sort(np.asarray(corners[0][:,0,0]))
    Xcorner-=np.min(Xcorner)
    dXcorner=np.diff(Xcorner)[0]
    Ycorner=np.sort(np.asarray(corners[1][0,:,0]))
    Ycorner-=np.min(Ycorner)
    dYcorner=np.diff(Ycorner)[0]
    Zcorner=np.sort(np.asarray(corners[2][0,0,:]))
    Zcorner-=np.min(Zcorner)
    dZcorner=np.diff(Zcorner)[0]

    ind_mask=np.asarray(np.where((LC_map>=Tbin_min)&(LC_map<Tbin_max)))
    Xmin=Xcorner[ind_mask[0]]
    Ymin=Ycorner[ind_mask[1]]
    Zmin=Zcorner[ind_mask[2]]

    mask_M=((lategrid[:,0]>=Xmin[0])&(lategrid[:,0]<Xmin[0]+dXcorner)&
            (lategrid[:,1]>=Ymin[0])&(lategrid[:,1]<Ymin[0]+dYcorner)&
            (lategrid[:,2]>=Zmin[0])&(lategrid[:,2]<Zmin[0]+dZcorner))
    for i in range(1,len(Xmin)):
        mask_M=mask_M|((lategrid[:,0]>=Xmin[i])&(lategrid[:,0]<Xmin[i]+dXcorner)&
                       (lategrid[:,1]>=Ymin[i])&(lategrid[:,1]<Ymin[i]+dYcorner)&
                       (lategrid[:,2]>=Zmin[i])&(lategrid[:,2]<Zmin[i]+dZcorner))

    M_in_Tbin=np.asarray((halos_in_survey_all['M_HALO'][filtering])[mask_M]*(LC_CO.Mpch.to(u.Mpc)))
    L_in_Tbin=np.asarray((LC_CO.halos_in_survey_all['CO_J10']['Lhalo'][filtering])[mask_M])
    return M_in_Tbin, L_in_Tbin


LC_CO=make_measurements(LC_params('CO'))
#CO_map=LC_CO.obs_3d_map

#bins = [4,8,16,27]
# -1sigma, 1sigma, 5sigma, 6sigma

#bin 4
#ang_lim=20*u.deg
#LC_CO.update(RAObs_width=ang_lim, DECObs_width=ang_lim)
#TiCO_edge=np.linspace(-4, 25, 50+1)
#M_in_T4, L_in_T4=halos_in_Tbin(LC_CO, TiCO_edge[4], TiCO_edge[4+1])
#np.save('/home/jlbernal/lim_LC_prods/M_in_Tbin4', M_in_T4)
#np.save('/home/jlbernal/lim_LC_prods/L_in_Tbin4', L_in_T4)

#bin 27
#ang_lim=15*u.deg
#LC_CO.update(RAObs_width=ang_lim, DECObs_width=ang_lim)
#TiCO_edge=np.linspace(-4, 25, 50+1)
#M_in_T27, L_in_T27=halos_in_Tbin(LC_CO, TiCO_edge[27], TiCO_edge[27+1])
#np.save('/home/jlbernal/lim_LC_prods/M_in_Tbin27', M_in_T27)
#np.save('/home/jlbernal/lim_LC_prods/L_in_Tbin27', L_in_T27)

#bin 8
#ang_lim=5*u.deg
#LC_CO.update(RAObs_width=ang_lim, DECObs_width=ang_lim)
#TiCO_edge=np.linspace(-4, 25, 50+1)
#M_in_T8, L_in_T8=halos_in_Tbin(LC_CO, TiCO_edge[8], TiCO_edge[8+1])
#np.save('/home/jlbernal/lim_LC_prods/M_in_Tbin8', M_in_T8)
#np.save('/home/jlbernal/lim_LC_prods/L_in_Tbin8', L_in_T8)

#bin 16
#ang_lim=8*u.deg
#LC_CO.update(RAObs_width=ang_lim, DECObs_width=ang_lim)
#TiCO_edge=np.linspace(-4, 25, 50+1)
#M_in_T16, L_in_T16=halos_in_Tbin(LC_CO, TiCO_edge[16], TiCO_edge[16+1])
#np.save('/home/jlbernal/lim_LC_prods/M_in_Tbin16', M_in_T16)
#np.save('/home/jlbernal/lim_LC_prods/L_in_Tbin16', L_in_T16)

#low 1%
ang_lim=14*u.deg
LC_CO.update(RAObs_width=ang_lim, DECObs_width=ang_lim)
#TiCO_edge=np.linspace(-4, 25, 50+1)
M_in_1p_low, L_in_1p_low=halos_in_Tbin(LC_CO, -4, -0.4325)
np.save('/home/jlbernal/lim_LC_prods/M_in_1p_low_small', M_in_1p_low)
np.save('/home/jlbernal/lim_LC_prods/L_in_1p_low_small', L_in_1p_low)

#mean 1%
#ang_lim=14*u.deg
#LC_CO.update(RAObs_width=ang_lim, DECObs_width=ang_lim)
#TiCO_edge=np.linspace(-4, 25, 50+1)
#M_in_1p_mean, L_in_1p_mean=halos_in_Tbin(LC_CO, -0.016,0.016)
#np.save('/home/jlbernal/lim_LC_prods/M_in_1p_mean_small', M_in_1p_mean)
#np.save('/home/jlbernal/lim_LC_prods/L_in_1p_mean_small', L_in_1p_mean)

#2sigma 1%
#ang_lim=14*u.deg
#LC_CO.update(RAObs_width=ang_lim, DECObs_width=ang_lim)
#TiCO_edge=np.linspace(-4, 25, 50+1)
#M_in_1p_2s, L_in_1p_2s=halos_in_Tbin(LC_CO, 2.125, 2.875)
#np.save('/home/jlbernal/lim_LC_prods/M_in_1p_2s_small', M_in_1p_2s)
#np.save('/home/jlbernal/lim_LC_prods/L_in_1p_2s_small', L_in_1p_2s)

#high 1%
#ang_lim=14*u.deg
#LC_CO.update(RAObs_width=ang_lim, DECObs_width=ang_lim)
#TiCO_edge=np.linspace(-4, 25, 50+1)
#M_in_1p_high, L_in_1p_high=halos_in_Tbin(LC_CO, 10.7, 140)
#np.save('/home/jlbernal/lim_LC_prods/M_in_1p_high_small', M_in_1p_high)
#np.save('/home/jlbernal/lim_LC_prods/L_in_1p_high_small', L_in_1p_high)
