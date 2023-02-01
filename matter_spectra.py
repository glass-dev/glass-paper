import glass.all

from _config import *

logger.info('shells:\n%s', shells)

logger.info('computing angular matter power spectra')

pars.Accuracy.TimeStepBoost = 5

lmax_ = max(lmax, lmax_gal)

logger.info('LMAX = %d', lmax_)

cls = glass.camb.matter_cls(pars, lmax_, weights, limber=False)

np.save(spec_path/'matter.npy', cls)
