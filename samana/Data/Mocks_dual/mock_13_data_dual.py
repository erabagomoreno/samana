import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data


class Mock13Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.6
        z_source = 1.6
        x_image = [-0.14252839,  0.16617677,  0.80976601, -0.70201045]
        y_image = [-1.12588213,  0.86375279,  0.30160594,  0.51056515]
        magnifications_true = [2.69778521, 7.05382693, 4.3565571 , 3.59387902]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.003307
        self.a4a_true = -0.00535
        self.delta_phi_m3_true = 0.29080
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock13Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
