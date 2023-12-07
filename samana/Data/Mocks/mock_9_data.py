import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_9_simple import image_data

class Mock9Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.66
        z_source = 2.7
        x_image = [ 0.20643186, -0.55304096, -0.94565392,  0.6539447 ]
        y_image = [ 1.2563651 , -0.90041679, -0.06573053, -0.36617873]
        magnifications_true = [2.89123265, 4.72612701, 4.32598561, 2.13596895]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0050
        self.a4a_true = 0.01754
        self.delta_phi_m3_true = -0.5127349
        self.delta_phi_m4_true = 0.0

        super(Mock9Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
