import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_3_simple_dual import image_data as simple_image_data

class Mock3Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False, cosmos_source_psf3=False):

        z_lens = 0.5
        z_source = 2.5
        x_image = [ 0.48120477, -0.47377756, -0.89419396,  0.66304533]
        y_image = [ 1.03149508, -0.89198088,  0.0698116 , -0.54534921]
        magnifications_true = [3.80307496, 5.87983207, 4.10676856, 4.32263284]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00514
        self.a4a_true = 0.01024
        self.delta_phi_m3_true = 0.053195
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock3Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)




