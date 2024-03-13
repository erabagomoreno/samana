import numpy as np
from samana.Model.multipole_prior_data import _vdis, _a3a, _a4a, _phi3, _phi4, _ellip
from scipy.stats import gaussian_kde


class OpticalMultipolePrior(object):
    """
    This class samples from the distribution of observed optical multipole moments presented by
    Hao et al. (2006) https://arxiv.org/pdf/astro-ph/0605319.pdf
    """
    def __init__(self):

        data = np.empty((5, len(_a3a)))
        data[0, :] = _a3a
        data[1, :] = _phi3
        data[2, :] = _a4a
        data[3, :] = _phi4
        q = 1 - _ellip
        data[4, :] = q
        weights = (_vdis / np.min(_vdis)) ** 4.0
        self._kde = gaussian_kde(data, weights=weights)

    def sample(self, q_mean=None, q_sigma=None):
        """
        Generate a sample from the joint distribution of multipole parameters measured by Hao et al.
        :param q_mean: optional argument specifying the axis ratio of a lens model; this well generate a sample from
        the conditional distribution p(a3, a4, phi3, phi4 | q)
        :param q_sigma: optional argument specifying the standard deviation of an inferred axis ratio of a lens model
        :return: a3_a, phi_3, a4_a, phi_4
        """
        if q_mean is None:
            sample = np.squeeze(self._kde.resample(1).T)
            (a3a, delta_phi_m3, a4a, delta_phi_m4, _) = sample
        else:
            while True:
                sample = np.squeeze(self._kde.resample(1).T)
                (a3a, delta_phi_m3, a4a, delta_phi_m4, q_draw) = sample
                prob = np.exp(-0.5 * (q_draw - q_mean)**2 / q_sigma**2)
                u = np.random.rand()
                if prob >= u:
                    break
        return a3a, delta_phi_m3, a4a, delta_phi_m4
