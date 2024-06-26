from samana.Data.data_base import QuadNoImageDataBase
import numpy as np


class _mock118(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):
        z_lens = 0.66
        z_source = 2.2
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_mock118, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)


class mock118(_mock118):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([-0.99143723,  1.01023381,  0.0968514 , -0.04121221])
        y_image = np.array([-0.40861256, -0.08531104, -0.93778356,  0.80972966])
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = None #ER: Not sure if I should include these
        magnifications = [2.76265764, 3.34008172, 2.24137652, 1.39867635] #ER: I dont include a second set of magnifications because I can do the summary statistic during the inference
        super(mock118, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                      flux_uncertainties,
                                      uncertainty_in_fluxes=False)
