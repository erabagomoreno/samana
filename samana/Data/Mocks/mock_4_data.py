import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_4_simple import image_data

class Mock4Data(MockBase):

    def __init__(self):

        z_lens = 0.5
        z_source = 1.5
        x_image = [ 0.26058618,  0.37927667,  1.00484952, -0.9052914 ]
        y_image = [-0.99863191,  1.00107117,  0.20801605, -0.01308953]
        magnifications_true = [5.2155646 , 5.40706899, 5.56605396, 2.98821292]
        magnification_measurement_errors = [ 0.00791124,  0.08109814, -0.16629849,  0.0621786 ]
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.005] * 4
        flux_uncertainties = [0.03] * 3

        self.a3a_true = 0.00225
        self.a4a_true = 0.000450
        self.delta_phi_m3_true = 0.4890
        self.delta_phi_m4_true = 0.0

        super(Mock4Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_uncertainties, image_data)
