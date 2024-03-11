# from samana.Data.data_base import ImagingDataBase
# import numpy as np
# from samana.Data.ImageData.he0435_814w import image_data, psf_error_map, psf_model
#
#
# class _HE0435(ImagingDataBase):
#
#     def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
#                  uncertainty_in_fluxes, supersample_factor=1.0):
#         z_lens =
#         z_source =
#         # we use all three flux ratios to constrain the model
#         keep_flux_ratio_index = [0, 1, 2]
#         self._psf_estimate_init = psf_model
#         self._psf_error_map_init = psf_error_map
#         self._image_data = image_data
#         self._supersample_factor = supersample_factor
#         image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
#         multi_band_list = [image_band]
#         kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
#         likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
#         super(_HE0435, self).__init__(z_lens, z_source,
#                                       kwargs_data_joint, x_image, y_image,
#                                       magnifications, image_position_uncertainties, flux_uncertainties,
#                                       uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
#                                       likelihood_mask_imaging_weights)
#
#     def likelihood_masks(self, x_image, y_image):
#         deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
#         _x = np.linspace(-window_size / 2, window_size / 2, image_data.shape[0])
#         _y = np.linspace(-window_size / 2, window_size / 2, image_data.shape[0])
#         _xx, _yy = np.meshgrid(_x, _y)
#         likelihood_mask = np.ones_like(_xx)
#         inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
#         likelihood_mask[inds] = 0.0
#         return likelihood_mask, likelihood_mask
#
#     @property
#     def kwargs_data(self):
#         _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
#         kwargs_data = {'background_rms':
#                            'exposure_time':
#         'ra_at_xy_0': ra_at_xy_0,
#         'dec_at_xy_0': dec_at_xy_0,
#         'transform_pix2angle': transform_pix2angle,
#         'image_data': self._image_data}
#         return kwargs_data
#
#     @property
#     def kwargs_numerics(self):
#         return {'supersampling_factor': int(self._supersample_factor),
#                 'supersampling_convolution': False}
#
#     @property
#     def coordinate_properties(self):
#         deltaPix =
#         window_size = *deltaPix
#         ra_at_xy_0 =
#         dec_at_xy_0 =
#         transform_pix2angle =
#         return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size
#
#     @property
#     def kwargs_psf(self):
#         kwargs_psf = {'psf_type': 'PIXEL',
#                       'kernel_point_source': self._psf_estimate_init,
#                       'psf_error_map': self._psf_error_map_init}
#         return kwargs_psf
#
#
# class HE0435_HST(_HE0435):
#
#     def __init__(self, supersample_factor=1.0):
#         """
#
#         :param image_position_uncertainties: list of astrometric uncertainties for each image
#         i.e. [0.003, 0.003, 0.003, 0.003]
#         :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
#         post-processing
#         :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
#         :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
#         """
#
#         flux_uncertainties =
#         super(HE0435_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
#                                          flux_uncertainties,
#                                          uncertainty_in_fluxes=True, supersample_factor=supersample_factor)
#
