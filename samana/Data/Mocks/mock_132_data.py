from samana.Data.data_base import QuadNoImageDataBase
import numpy as np


class _mock132(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):
        z_lens = 0.5
        z_source = 2.5
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_mock132, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)


class mock132(_mock132):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 1.02395552, -0.94917023, -0.13830323, -0.0893606 ])
        y_image = np.array([ 0.38242424,  0.23700839,  0.9090916 , -0.79549461])
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = None #ER: Not sure if I should include these
        magnifications = [4.50273953, 6.23404872, 5.3059758 , 3.1430919 ] #ER: I dont include a second set of magnifications because I can do the summary statistic during the inference
        super(mock132, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                      flux_uncertainties,
                                      uncertainty_in_fluxes=False)
