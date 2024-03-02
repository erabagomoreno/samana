from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _J2038(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        z_lens = 0.23
        z_source = 0.78
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_J2038, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class J2038JWST(_J2038):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 0.71909739,  0.84809739, -0.66290261, -1.45890261])
        y_image = np.array([ 0.88371611, -1.20728389, -1.17828389,  0.49971611])
        image_position_uncertainties = [0.005] * 4 # 5 marcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J2038JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)

    @property
    def coordinate_properties(self):
        window_size = 6.0
        deltaPix = 0.05
        ra_at_xy_0 = -3.0
        dec_at_xy_0 = -3.0
        transform_pix2angle = np.array([[0.05, 0.], [0., 0.05]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size
