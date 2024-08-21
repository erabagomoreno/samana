import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_5_simple_dual import image_data as simple_image_data

class Mock5Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.4
        z_source = 1.6
        x_image = [-1.05585635,  0.91992255, -0.23825644,  0.5460858 ]
        y_image = [ 0.45087947, -0.45346447, -0.83582192,  0.68217662]
        magnifications_true = [2.62364263, 3.70275796, 2.43672701, 2.23769903]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00146
        self.a4a_true = 0.00371
        self.delta_phi_m3_true = -0.2911280
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock5Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


