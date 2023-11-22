import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_3_simple import image_data

class Mock3Data(MockBase):

    def __init__(self):

        z_lens = 0.5
        z_source = 2.5
        x_image = [ 0.26197365,  0.37192752,  0.99587974, -0.89833527]
        y_image = [-0.98968167,  0.99370493,  0.21072101, -0.01553163]
        magnifications_true = [5.1207985 , 5.22847917, 5.41861705, 2.92272106]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00514
        self.a4a_true = 0.01024
        self.delta_phi_m3_true = 0.053195
        self.delta_phi_m4_true = 0.0

        super(Mock3Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data)
