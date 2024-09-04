import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data

class Mock18Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.52
        z_source = 1.82
        x_image = [-0.97957209,  0.88273349, -0.27556891,  0.53652596]
        y_image = [ 0.43337834, -0.54482013, -0.87575254,  0.73678855]
        magnifications_true = [6.41808498, 8.7576207 , 7.76147606, 6.41634476]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0052722
        self.a4a_true = 0.000151
        self.delta_phi_m3_true = 0.1574715
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock18Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
