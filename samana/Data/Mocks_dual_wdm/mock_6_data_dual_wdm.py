import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_6_simple_dual import image_data as simple_image_data


class Mock6Data_dual_wdm(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.55
        z_source = 2.6
        x_image = [ 0.61140625, -0.15268665,  0.65883327, -0.76532536]
        y_image = [-0.90723964,  0.95312052,  0.6564056 , -0.22731122]
        magnifications_true = [3.04682008, 5.17932949, 3.44212968, 2.05592872]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00502
        self.a4a_true = -0.005256
        self.delta_phi_m3_true = 0.411402
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock6Data_dual_wdm, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


