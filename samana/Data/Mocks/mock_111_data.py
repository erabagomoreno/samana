from samana.Data.data_base import QuadNoImageDataBase
import numpy as np


class _mock111(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):
        z_lens = 0.45
        z_source = 3.1
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_mock111, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                     flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)


class mock111(_mock111):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([-0.24029732, 0.09637579, -0.86122488, 0.90729624])
        y_image = np.array([1.02987655, -0.95124605, -0.31704983, -0.06750402])
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = None #ER: Not sure if I should include these
        magnifications = [3.66, 6.35, 3.66, 3.07] #ER: I dont include a second set of magnifications because I can do the summary statistic during the inference
        super(mock111, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                        flux_uncertainties,
                                        uncertainty_in_fluxes=False)
