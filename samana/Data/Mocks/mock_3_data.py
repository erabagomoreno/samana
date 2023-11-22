import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_3_simple import image_data

class Mock3Data(MockBase):

    def __init__(self):

        z_lens = 0.5
        z_source = 2.0
        x_image = [ 0.48314461, -0.54340102, -0.94422129,  0.68977071]
        y_image = [ 1.09978087, -0.89038174,  0.0485766 , -0.54368415]
        magnifications_true = [3.99566627, 6.91309483, 4.62162504, 4.0917622 ]
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
