import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_8_simple import image_data

class Mock8Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.5
        z_source = 1.6
        x_image = [-0.89373101,  0.63411634, -0.62676762,  0.65904621]
        y_image = [-0.62638899,  0.85352402,  0.73662769, -0.63904648]
        magnifications_true = [5.72468493, 7.90861551, 6.8862955 , 4.50563578]
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
