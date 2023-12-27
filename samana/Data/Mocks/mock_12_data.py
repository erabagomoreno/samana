import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_12_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_12_cosmos import image_data as cosmos_image_data


class Mock12Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.55
        z_source = 2.15
        x_image = [ 1.14970621, -0.84741026, -0.3033323 , -0.19716524]
        y_image = [ 0.25812294,  0.63820593,  0.95162651, -0.77428388]
        magnifications_true = [ 4.70188868, 10.95124582,  9.55649461,  2.18874825]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004408
        self.a4a_true = 0.00894
        self.delta_phi_m3_true = -0.3621598
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            image_data = simple_image_data
        super(Mock12Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
