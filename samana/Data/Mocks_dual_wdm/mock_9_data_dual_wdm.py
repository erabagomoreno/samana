import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_9_simple_dual import image_data as simple_image_data


class Mock9Data_dual_wdm(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.66
        z_source = 2.7
        x_image = [ 0.14941407, -0.51661153, -0.89339207,  0.62851627]
        y_image = [ 1.19457561, -0.84894688, -0.13408813, -0.36569291]
        magnifications_true = [2.87133489, 5.18773276, 4.87394101, 2.10983042]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.005
        self.a4a_true = -0.0395
        self.delta_phi_m3_true = -0.5127
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock9Data_dual_wdm, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

