import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_6_simple import image_data

class Mock7Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.35
        z_source = 1.4
        x_image = [-0.16234706,  0.98575445,  0.80004301, -0.55537481]
        y_image = [ 1.0989313 , -0.36710543,  0.65941259, -0.47587784]
        magnifications_true = [3.41113138, 4.05698314, 3.19503437, 1.14192612]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.001603
        self.a4a_true = 0.0145708
        self.delta_phi_m3_true = -0.443688
        self.delta_phi_m4_true = 0.0

        super(Mock7Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
