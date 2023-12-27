import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_13_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_13_cosmos import image_data as cosmos_image_data


class Mock13Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.6
        z_source = 1.6
        x_image = [-0.17869849,  0.16789957,  0.85755153, -0.7104615 ]
        y_image = [-1.13534269,  0.88977598,  0.25366589,  0.55462529]
        magnifications_true = [2.88019183, 10.6311993,   4.3069418,   4.54614156]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00330
        self.a4a_true = -0.005354
        self.delta_phi_m3_true = 0.290809
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            image_data = simple_image_data
        super(Mock13Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
