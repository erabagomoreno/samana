import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_4_simple_dual import image_data as simple_image_data

class Mock4Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.5
        z_source = 1.5
        x_image = [ 0.28055407,  0.34591097,  0.95996187, -0.87343163]
        y_image = [-0.96895397,  0.96808478,  0.25710428, -0.03480812]
        magnifications_true = [4.72458142, 5.0417822 , 5.07262986, 2.68080343]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00225
        self.a4a_true = 0.000450
        self.delta_phi_m3_true = 0.4890
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock4Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


