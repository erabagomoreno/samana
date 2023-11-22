import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_5_simple import image_data

class Mock5Data(MockBase):

    def __init__(self):

        z_lens = 0.65
        z_source = 2.7
        x_image = [-0.70988483,  0.47758486, -0.48481062,  0.88986211]
        y_image = [ 0.84036997, -0.87221836, -0.83707619,  0.28380713]
        magnifications_true = [4.46849783, 7.04000934, 5.62408315, 5.42222398]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00146
        self.a4a_true = 0.00371
        self.delta_phi_m3_true = -0.2911280
        self.delta_phi_m4_true = 0.0

        super(Mock5Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data)
