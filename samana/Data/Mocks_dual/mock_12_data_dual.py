import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data

class Mock12Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.55
        z_source = 2.15
        x_image = [ 1.11630686, -0.92268147, -0.15778883, -0.18481431]
        y_image = [ 0.16870915,  0.46584558,  0.92357915, -0.78849725]
        magnifications_true = [4.49587928, 6.93224524, 6.34388067, 2.81398224]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004408
        self.a4a_true = 0.00894
        self.delta_phi_m3_true = -0.3621598
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock12Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
