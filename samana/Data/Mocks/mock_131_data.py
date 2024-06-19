from samana.Data.data_base import QuadNoImageDataBase
import numpy as np


class _mock131(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):
        z_lens = 0.4
        z_source = 1.7
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_mock131, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)


class mock131(_mock131):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 0.16522581,  0.30413598,  0.8518448 , -0.63773724])
        y_image = np.array([ 1.26423357, -0.98610978, -0.12679536, -0.15436398])
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = None #ER: Not sure if I should include these
        magnifications = [2.36988211, 3.59185755, 2.20424975, 1.2078695 ] #ER: I dont include a second set of magnifications because I can do the summary statistic during the inference
        super(mock131, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                      flux_uncertainties,
                                      uncertainty_in_fluxes=False)
