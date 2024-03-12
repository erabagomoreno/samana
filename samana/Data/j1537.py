from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _J1537(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        #z_lens = 0.592
        z_lens = 0.6
        #z_source = 1.721
        z_source = 1.7
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_J1537, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class J1537JWST(_J1537):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array( [ 1.42722809, -0.56577191, -1.42077191,  0.67722809])
        y_image = np.array( [-0.71655342, -1.04555342,  0.92744658,  1.04644658])
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = [0.02/0.73, 0.02/0.94, 0.02/0.73]
        magnifications = np.array([1.0, 0.73, 0.94, 0.73])
        super(J1537JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                                uncertainty_in_fluxes=False)

    @property
    def coordinate_properties(self):
        window_size = 8.0
        deltaPix = 0.05
        ra_at_xy_0 = -4.0
        dec_at_xy_0 = -4.0
        transform_pix2angle = np.array([[0.05, 0.], [0., 0.05]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size
