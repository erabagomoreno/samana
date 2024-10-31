import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_10_simple_dual import image_data as simple_image_data


class Mock10Data_dual_wdm(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.45
        z_source = 3.1
        x_image = [-0.74538264,  0.78339438,  0.70799406, -0.49272041]
        y_image = [ 0.7994885 , -0.62365477,  0.53537562, -0.6904502 ]
        magnifications_true = [3.67380926, 4.75868475, 3.28195108, 2.8620814 ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0001888
        self.a4a_true = -0.0013544
        self.delta_phi_m3_true = 0.28412
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock10Data_dual_wdm, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

