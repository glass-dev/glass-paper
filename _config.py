import os
from pathlib import Path
import logging
import numpy as np
from cosmology import Cosmology

# use CAMB for cosmology and angular power spectra
# many other choices are possible!
import camb

import glass.all


# parameters of the simulation
nside = 4096
lmax = 5000

# maximum redshift of the simulation
zend = 2.

# number of correlated matter fields
ncorr = 5

# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# random number generator, fix seed to make reproducible
rng = np.random.default_rng()

# which individual redshifts to show
showz = [0.5, 1.0, 2.0]

# redshift array for various functions
zz = np.linspace(0, zend, 1000)

# where spectra are stored
spec_path = Path('spectra')

# where data files are stored
data_path = Path('data')

# where plots are stored
plot_path = Path('plot')


# CAMB config
# -----------

# set up and store CAMB parameters
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                       NonLinear=camb.model.NonLinear_both)

# this is the cosmology object used in the benchmarks
cosmo = Cosmology.from_camb(pars)


# matter
# ------

# shell setup; here equal thickness of 150 Mpc in comoving distance
shells = glass.matter.distance_shells(cosmo, 0., zend, dx=150.)

# matter weights; the zlin is necessary for CAMB to produce good results
weights = glass.matter.uniform_weights(shells, zlin=0.1)


# galaxies
# --------

# settings for galaxy analysis
nside_gal = 2048
lmax_gal = 3000

# number of correlated matter fields for galaxy analysis
ncorr_gal = 5

# angular mode binning;  set to None for no binning
gal_lbin = np.unique(np.geomspace(2, lmax_gal, 41, dtype=int))

# default bias parameter
# this is less than unity so that the density will not get clipped
# which would require non-linear theory bias computation
gal_bias = 0.8

# photometric redshift distribution
gal_mean_z = [0.5, 1.0]
gal_sigma_z = 0.125

# total number of galaxies per arcmin2 in each bin
gal_dens = 1.0

# galaxy number density in units of galaxies/arcmin2/dz
dndz = glass.observations.gaussian_nz(zz, gal_mean_z, gal_sigma_z, norm=gal_dens)

# the number of galaxy populations (= bins)
nbin = dndz.shape[0]


# job config
# ----------

# this is to give multiple runs a different output filename
if 'SLURM_JOB_ID' in os.environ:
    job_id = os.environ['SLURM_JOB_ID']
else:
    job_id = os.getpid()


# logging
# -------

logger = logging.getLogger('glass')
loglevel = logging.INFO
loghandler = logging.StreamHandler()
logformatter = logging.Formatter('%(message)s')
loghandler.setFormatter(logformatter)
loghandler.setLevel(loglevel)
logger.addHandler(loghandler)
logger.setLevel(loglevel)
