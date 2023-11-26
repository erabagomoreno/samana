import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_6_simple import image_data

class Mock6Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.55
        z_source = 2.6
        x_image = [ 0.62623824, -0.23244766,  0.7292022 , -0.80853096]
        y_image = [-0.91953787,  0.97866734,  0.62435675, -0.30273136]
        magnifications_true = [4.07148687, 4.354839  , 3.40278675, 2.3364734 ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00502
        self.a4a_true = -0.005256
        self.delta_phi_m3_true = 0.411402
        self.delta_phi_m4_true = 0.0

        super(Mock6Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
