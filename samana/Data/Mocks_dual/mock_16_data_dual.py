import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data

class Mock16Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.33
        z_source = 0.9
        x_image = [-1.113174  ,  0.51445076, -0.2591932 ,  0.43776403]
        y_image = [-0.11639938,  0.91001102,  0.94754684, -0.67224785]
        magnifications_true = [4.35798223, 8.67576247, 8.08837432, 1.97884512]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.013472
        self.a4a_true = 0.01230
        self.delta_phi_m3_true = -0.28976
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock16Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
