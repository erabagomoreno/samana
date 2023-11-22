import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_1_simple import image_data

class Mock1Data(MockBase):

    def __init__(self):

        z_lens = 0.5
        z_source = 2.2
        x_image = [-0.92161217,  0.78024418, -0.25056735,  0.70611926]
        y_image = [-0.6914899 ,  0.61451654,  0.86960051, -0.52450815]
        magnifications_true = [ 4.34866761, 10.45921158,  4.45857144,  4.52709824]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004010
        self.a4a_true = -0.004488
        self.delta_phi_m3_true = -0.08689
        self.delta_phi_m4_true = 0.0

        super(Mock1Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data)
