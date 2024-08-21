import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data


class Mock1Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False,
                 cosmos_source_psf3=False):

        z_lens = 0.5
        z_source = 2.2
        x_image = [-0.88458412,  0.72590717, -0.26512332,  0.69475691]
        y_image = [-0.67755559,  0.63728695,  0.85135217, -0.4898624 ]
        magnifications_true = [4.57823424, 7.13481907, 4.78367834, 4.3470874 ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004010
        self.a4a_true = -0.004488
        self.delta_phi_m3_true = -0.08689
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock1Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                        image_data, super_sample_factor)



