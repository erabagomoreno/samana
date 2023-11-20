import numpy as np

class ImagingDataBase(object):

    def __init__(self, zlens, zsource, kwargs_data_joint, x_image, y_image,
                 magnifications, image_position_uncertainties, flux_uncertainty,
                 uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask, likelihood_mask_imaging_weights):

        self.z_lens = zlens
        self.z_source = zsource
        self._kwargs_data_joint = kwargs_data_joint
        self._x_image_init = np.array(x_image)
        self._y_image_init = np.array(y_image)
        self._x = np.array(x_image)
        self._y = np.array(y_image)
        self.magnifications = magnifications
        self.image_position_uncertainty = image_position_uncertainties
        if len(self.image_position_uncertainty) != len(self._x_image_init):
            raise Exception('image position uncertainties must have the same shape as point source arrays')
        self.flux_uncertainty = flux_uncertainty
        self.uncertainty_in_fluxes = uncertainty_in_fluxes
        self.keep_flux_ratio_index = keep_flux_ratio_index
        self.likelihood_mask = likelihood_mask
        self.likelihood_mask_imaging_weights = likelihood_mask_imaging_weights

    def perturb_image_positions(self, delta_x_image=None, delta_y_image=None):
        if delta_x_image is None:
            delta_x_image = np.random.normal(0.0, self.image_position_uncertainty)
        if delta_y_image is None:
            delta_y_image = np.random.normal(0.0, self.image_position_uncertainty)
        self._x = self._x_image_init + delta_x_image
        self._y = self._y_image_init + delta_y_image
        return delta_x_image, delta_y_image

    @property
    def x_image(self):
        return self._x

    @property
    def y_image(self):
        return self._y

    @property
    def coordinate_properties(self):
        raise Exception('must define a coordinate_properties property in the data class')

    @property
    def kwargs_data_joint(self):
        return self._kwargs_data_joint

    @property
    def kwargs_psf(self):
        raise Exception('must define a kwargs_psf property in the data class')

    @property
    def kwargs_numerics(self):
        raise Exception('must define a kwargs_numerics property in the data class')
