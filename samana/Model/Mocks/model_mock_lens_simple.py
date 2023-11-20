from samana.Model.model_base import ModelBase
import numpy as np


class MockModelBase(ModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None):
        self._shapelets_order = shapelets_order
        super(MockModelBase, self).__init__(data_class, kde_sampler)

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.2], [0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005]]

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [{'amp': 20.0, 'R_sersic': 0.3, 'n_sersic': 4., 'e1': -0.0,
                                   'e2': 0.05, 'center_x': 0.0, 'center_y': 0.0}]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.01, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.25, 'center_y': -0.25}]
        kwargs_upper_lens_light = [
            {'R_sersic': 5, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.25, 'center_y': 0.25}]
        kwargs_lens_light_fixed = [{}]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed,
                             kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        return lens_light_model_list, lens_light_params

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': True,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_likelihood': False,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        kwargs_constraints = {'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True
                              }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints
