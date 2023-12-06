import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_8_simple import image_data

class Mock9Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.66
        z_source = 2.7
        x_image = [ 0.5734881 , -1.01091562, -0.79187125,  0.48254545]
        y_image = [ 1.06767472, -0.35142928,  0.59087936, -0.59761061]
        magnifications_true = [3.02775716, 5.81482074, 4.13887137, 1.63059334]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.019782
        self.a4a_true = 0.01754
        self.delta_phi_m3_true = -0.5127349
        self.delta_phi_m4_true = 0.0

        super(Mock9Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
