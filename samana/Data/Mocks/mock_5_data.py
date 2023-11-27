import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_5_simple import image_data

class Mock5Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.4
        z_source = 1.6
        x_image = [-1.04814384,  0.99262214,  0.50969389, -0.30372116]
        y_image = [ 0.47660129, -0.42046054,  0.73688046, -0.8312197 ]
        magnifications_true = [3.06880609, 3.54699025, 2.34253429, 2.36118715]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00146
        self.a4a_true = 0.00371
        self.delta_phi_m3_true = -0.2911280
        self.delta_phi_m4_true = 0.0

        super(Mock5Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
