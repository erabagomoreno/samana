import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_2_simple_dual import image_data as simple_image_data

class Mock2Data_dual_wdm(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False, cosmos_source_psf3=False):

        z_lens = 0.4
        z_source = 1.7
        x_image = [ 0.91873143, -0.8661518 , -0.71845576,  0.1432804 ]
        y_image = [ 0.7601082 , -0.23025655,  0.47253701, -0.78011722]
        magnifications_true = [2.87721104, 6.99528403, 5.56837358, 2.32931165]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0002277
        self.a4a_true = -0.004348
        self.delta_phi_m3_true = -0.067025
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock2Data_dual_wdm, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

