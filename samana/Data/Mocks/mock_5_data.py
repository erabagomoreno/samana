import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_5_simple import image_data

class Mock5Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.45
        z_source = 1.6
        x_image = [-0.62164854,  0.28915459, -0.74308497,  0.82045643]
        y_image = [ 0.92849105, -1.01651516, -0.59884716,  0.32875051]
        magnifications_true = [3.46577852, 4.45480934, 3.49966457, 2.50003242]
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
