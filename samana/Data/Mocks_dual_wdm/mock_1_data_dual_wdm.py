import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data


class Mock1Data_dual_wdm(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False,
                 cosmos_source_psf3=False):

        z_lens = 0.5
        z_source = 2.2
        x_image = [-0.90794793,  0.77190002,  0.7212967 , -0.23534781]
        y_image = [-0.69635551,  0.59460802, -0.47857989,  0.86566705]
        magnifications_true = [3.98858286, 7.35510345, 4.8094685 , 4.51146317]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004010
        self.a4a_true = -0.004488
        self.delta_phi_m3_true = -0.08689
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock1Data_dual_wdm, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                        image_data, super_sample_factor)



