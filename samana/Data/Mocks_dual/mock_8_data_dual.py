import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_8_simple_dual import image_data as simple_image_data



class Mock8Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.5
        z_source = 1.6
        x_image = [-0.90583206,  0.46948266, -0.55082004,  0.67124002]
        y_image = [-0.55553276,  0.90458441,  0.76434735, -0.56306828]
        magnifications_true = [5.29640558, 7.87083145, 6.9083089 , 4.21069207]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00782
        self.a4a_true = 0.001805
        self.delta_phi_m3_true = 0.3910
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock8Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)



