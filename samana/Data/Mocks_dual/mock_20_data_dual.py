import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data

class Mock20Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.35
        z_source = 2.1
        x_image = [ 0.02534674,  0.5607042 ,  0.9197616 , -0.73319956]
        y_image = [-1.14436637,  0.82095574,  0.24813988,  0.24982744]
        magnifications_true = [3.30033239, 6.08480048, 4.69475556, 2.20028057]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004752
        self.a4a_true = 0.015022
        self.delta_phi_m3_true = 0.0922903
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock20Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
