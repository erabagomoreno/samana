import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_1_simple import image_data

class Mock1Data(MockBase):

    def __init__(self):

        z_lens = 0.5
        z_source = 2.2
        x_image = [-0.93265855,  0.78855685, -0.25439243,  0.71549713]
        y_image = [-0.70236205,  0.62673126,  0.88038215, -0.52959132]
        magnifications_true = [4.42309835, 11.66150862, 4.57630222, 4.64745929]
        magnification_measurement_errors = [0.21553918, -0.21402008, -0.07251221, -0.14959734]
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.005] * 4
        flux_ratio_uncertainties = [0.03] * 3

        self.a3a_true = -0.004010
        self.a4a_true = -0.004488
        self.delta_phi_m3_true = -0.08689
        self.delta_phi_m4_true = 0.0

        super(Mock1Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data)
