from samana.Data.data_base import QuadNoImageDataBase
import numpy as np


class _mock114(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):
        z_lens = 0.55
        z_source = 2.15
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_mock114, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                     flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)


class mock114(_mock114):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([1.07675419, -1.0132381, -0.48203367, 0.17010669])
        y_image = np.array([0.41587661, -0.12956416, 0.76073129, -0.78687879])
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = None #ER: Not sure if I should include these
        magnifications = [2.69, 3.78, 2.37, 1.93] #ER: I dont include a second set of magnifications because I can do the summary statistic during the inference
        super(mock114, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                        flux_uncertainties,
                                        uncertainty_in_fluxes=False)
