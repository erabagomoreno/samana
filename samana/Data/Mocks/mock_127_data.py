from samana.Data.data_base import QuadNoImageDataBase
import numpy as np


class _mock127(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):
        z_lens = 0.3
        z_source = 1.45
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_mock127, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)


class mock127(_mock127):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 0.72923929, -0.47809199,  0.5493901 , -0.78522729])
        y_image = np.array([ 0.85131458, -0.89396549, -0.72309467,  0.3456308 ])
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = None #ER: Not sure if I should include these
        magnifications = [4.58572728, 7.45806486, 5.65471284, 4.12101551] #ER: I dont include a second set of magnifications because I can do the summary statistic during the inference
        super(mock127, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                      flux_uncertainties,
                                      uncertainty_in_fluxes=False)
