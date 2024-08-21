import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_9_simple_dual import image_data as simple_image_data


class Mock9Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.66
        z_source = 2.7
        x_image = [ 0.22686688, -0.43355021, -0.87080907,  0.64295654]
        y_image = [ 1.20340353, -0.90065071, -0.06191534, -0.37333653]
        magnifications_true = [2.71909502, 4.46454784, 3.84907776, 2.10078735]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.005
        self.a4a_true = -0.0395
        self.delta_phi_m3_true = -0.5127
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock9Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

