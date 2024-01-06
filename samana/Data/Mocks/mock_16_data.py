import numpy as np
from samana.Data.Mocks.base import MockBase
#from samana.Data.ImageData.mock_16_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_16_cosmos import image_data as cosmos_image_data


class Mock16Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.33
        z_source = 0.9
        x_image = [-1.13845796,  0.51598964, -0.33517658,  0.4351658 ]
        y_image = [-0.10259492,  0.94280777,  0.95155293, -0.68814576]
        magnifications_true = [4.48103259, 8.16400184, 8.00162568, 2.03547555]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.013472
        self.a4a_true = 0.01230
        self.delta_phi_m3_true = -0.28976
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock16Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
