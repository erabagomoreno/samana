import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data


class Mock14Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.3
        z_source = 1.8
        x_image = [-1.13117692,  0.89279211,  0.05023512,  0.30560756]
        y_image = [-0.03849772,  0.58553966,  0.94263652, -0.75804144]
        magnifications_true = [4.38032249, 6.63134481, 6.4952334 , 2.69649061]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00231
        self.a4a_true = 0.0034161
        self.delta_phi_m3_true = 0.014601
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock14Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
