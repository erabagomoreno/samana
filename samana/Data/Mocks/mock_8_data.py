import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_6_simple import image_data

class Mock8Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.3
        z_source = 0.9
        x_image = [ 0.78299232, -0.76274888, -0.79047552,  0.44990524]
        y_image = [ 0.80338065, -0.70578691,  0.4880058 , -0.80951851]
        magnifications_true = [4.97427634, 7.10939417, 5.05693453, 5.23806595]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00782
        self.a4a_true = 0.001805
        self.delta_phi_m3_true = 0.3910
        self.delta_phi_m4_true = 0.0

        super(Mock8Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
