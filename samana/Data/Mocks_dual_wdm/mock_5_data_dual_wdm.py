import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_5_simple_dual import image_data as simple_image_data

class Mock5Data_dual_wdm(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.4
        z_source = 1.6
        x_image = [-1.04725877,  0.92831927, -0.26242184,  0.53372054]
        y_image = [ 0.45106902, -0.46176207, -0.83137052,  0.69377402]
        magnifications_true = [2.68484266, 3.62593081, 2.35831116, 2.20276456]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00146
        self.a4a_true = 0.00371
        self.delta_phi_m3_true = -0.2911280
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock5Data_dual_wdm, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


