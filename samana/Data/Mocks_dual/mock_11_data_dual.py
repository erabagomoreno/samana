import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data

class Mock11Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.6
        z_source = 1.9
        x_image = [ 0.98077291, -0.3379609 ,  0.65135508, -0.54445908]
        y_image = [-0.49237469,  1.04039908,  0.74046044, -0.51777584]
        magnifications_true = [3.82826363, 4.21015452, 4.43715567, 1.68310085]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00014642
        self.a4a_true = -0.003825
        self.delta_phi_m3_true = -0.3348207
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock11Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
