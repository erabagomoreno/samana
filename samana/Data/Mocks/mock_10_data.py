import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_10_simple import image_data

class Mock10Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.6
        z_source = 3.1
        x_image = [-0.7466955 ,  0.91697486,  0.62837121, -0.58124281]
        y_image = [ 0.82217278, -0.60594587,  0.72709186, -0.65363871]
        magnifications_true = [4.35580508, 4.39092013, 3.41460735, 2.64781887]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0001888
        self.a4a_true = -0.0013544
        self.delta_phi_m3_true = 0.28412
        self.delta_phi_m4_true = 0.0

        super(Mock10Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
