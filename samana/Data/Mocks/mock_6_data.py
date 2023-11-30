import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_6_simple import image_data

class Mock6Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.3
        z_source = 1.6
        x_image = [ 0.51425757, -0.53602578, -0.74904253,  0.68241391]
        y_image = [-1.00447728,  0.92292458, -0.38746628,  0.47857275]
        magnifications_true = [3.66949542, 4.08239418, 2.55090467, 2.46385224]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00502458
        self.a4a_true = -0.005256
        self.delta_phi_m3_true = 0.411402
        self.delta_phi_m4_true = 0.0

        super(Mock6Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
