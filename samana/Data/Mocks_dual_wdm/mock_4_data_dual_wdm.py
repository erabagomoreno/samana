import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_4_simple_dual import image_data as simple_image_data

class Mock4Data_dual_wdm(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.5
        z_source = 1.5
        x_image = [ 0.26394069,  0.32066376,  0.95391088, -0.87920986]
        y_image = [-0.97883003,  0.97000047,  0.30044041, -0.01943691]
        magnifications_true = [4.55726078, 5.38071542, 5.54885674, 2.84635324]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00225
        self.a4a_true = 0.000450
        self.delta_phi_m3_true = 0.4890
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock4Data_dual_wdm, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


