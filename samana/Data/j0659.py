from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _J0659(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        z_lens = 0.77
        z_source = 3.1
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_J0659, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class J0659JWST(_J0659):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 1.86868455, -2.79631545,  0.88968455,  1.95268455])
        y_image = np.array([-0.92780858, -1.26280858,  1.96419142,  0.97519142])
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J0659JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)

    @property
    def coordinate_properties(self):
        window_size = 8.0
        deltaPix = 0.05
        ra_at_xy_0 = -4.0
        dec_at_xy_0 = -4.0
        transform_pix2angle = np.array([[0.05, 0.], [0., 0.05]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size
