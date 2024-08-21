import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_6_simple_dual import image_data as simple_image_data


class Mock6Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.55
        z_source = 2.6
        x_image = [ 0.61319989, -0.16837857,  0.6783684 , -0.76361477]
        y_image = [-0.88213815,  0.95340541,  0.63408901, -0.25116101]
        magnifications_true = [3.22450138, 4.39780802, 3.31922952, 2.04211829]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00502
        self.a4a_true = -0.005256
        self.delta_phi_m3_true = 0.411402
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock6Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


