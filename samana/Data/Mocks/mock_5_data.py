import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_4_simple import image_data

class Mock5Data(MockBase):

    def __init__(self):

        z_lens = 0.6
        z_source = 2.0
        x_image = [ 0.91448269, -0.02065917,  0.75902066, -0.7014605 ]
        y_image = [-0.60258736,  1.09147773,  0.68361138, -0.40694685]
        magnifications_true =  [4.26083207, 5.41125475, 5.63482828, 1.81065822]
        magnification_measurement_errors = [ 0.05639989, -0.05371268,  0.41090935, -0.01369358]
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.005] * 4
        flux_uncertainties = [0.03] * 3

        self.a3a_true = -0.00146
        self.a4a_true = 0.00371
        self.delta_phi_m3_true = -0.2911280
        self.delta_phi_m4_true = 0.0

        super(Mock5Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_uncertainties, image_data)
