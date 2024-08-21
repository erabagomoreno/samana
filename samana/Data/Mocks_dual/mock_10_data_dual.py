import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_10_simple_dual import image_data as simple_image_data


class Mock10Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.45
        z_source = 3.1
        x_image = [-0.81729593,  0.76522115,  0.71649052, -0.46463292]
        y_image = [ 0.7741974 , -0.65616896,  0.52359727, -0.73178927]
        magnifications_true = [3.42369734, 4.92094966, 3.03670965, 3.15055975]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0001888
        self.a4a_true = -0.0013544
        self.delta_phi_m3_true = 0.28412
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock10Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

