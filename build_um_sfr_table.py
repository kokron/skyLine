import numpy as np
from glob import glob
import time
import gc
#Directory goes here
lcdir = '/home/users/kokron/scratch/MDPL2-UM/'
dtype = np.dtype(dtype=[('id', 'i8'),('descid','i8'),('upid','i8'),
                        ('flags', 'i4'), ('uparent_dist', 'f4'),
                        ('pos', 'f4', (6)), ('vmp', 'f4'), ('lvmp', 'f4'),
                        ('mp', 'f4'), ('m', 'f4'), ('v', 'f4'), ('r', 'f4'),
                        ('rank1', 'f4'), ('rank2', 'f4'), ('ra', 'f4'),
                        ('rarank', 'f4'), ('A_UV', 'f4'), ('sm', 'f4'),
                        ('icl', 'f4'), ('sfr', 'f4'), ('obs_sm', 'f4'),
                        ('obs_sfr', 'f4'), ('obs_uv', 'f4'), ('empty', 'f4')],
                 align=True)

behroozi_sfr = np.loadtxt('../SFR_tables/sfr_table_Behroozi.dat')
mcenters = np.unique(behroozi_sfr[:,1])

dm = 0.05

fnames = np.sort(glob(lcdir+'sfr_*'))

sublabels= ['m', 'sfr']

sfrtable = np.zeros(shape=(len(fnames)*len(mcenters), 4))


# bigcat = np.array(a[1].data)
start_time = time.time()




for i, fname in enumerate(fnames):
    snap =  np.fromfile(fname, dtype=dtype)    

    subsnap = snap[sublabels]

    del snap
    gc.collect()

    logems = np.log10(subsnap['m'])

    snapscale = float(fname.split('_')[-1].split('bin')[0][:-1])
    
    snapz = 1./snapscale - 1

    for j, mcenter in enumerate(mcenters):

        tableidx = i*len(mcenters) + j
        msubidx = (logems < mcenter + dm*0.5)&(logems > mcenter - dm*0.5)

        msub = subsnap['m'][msubidx]
        sfrsub = subsnap['sfr'][msubidx]

        logsfr = np.log10(sfrsub)

        finitesfr = logsfr[np.isfinite(logsfr)]

    #     medsfr = np.median(sfrsub)

        meanlogsfr = np.mean(finitesfr)

        stdlogsfr = np.std(finitesfr)

        sfrtable[tableidx,0] = snapz
        sfrtable[tableidx,1] = mcenter
        sfrtable[tableidx,2] = meanlogsfr
        sfrtable[tableidx,3] = stdlogsfr
    print(i, time.time() - start_time)
    if i == 30:
        brak
np.savetxt('../SFR_tables/sfr_table_UniverseMachine.dat')

