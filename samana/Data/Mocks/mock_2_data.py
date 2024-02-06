import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_2_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_2_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_2_2038 import image_data as simulated_2038_image_data

class Mock2Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.4
        z_source = 1.7
        x_image = [ 0.96123035, -0.88968036, -0.70396572,  0.15753392]
        y_image = [ 0.76392051, -0.2642994 ,  0.53126911, -0.80157403]
        magnifications_true = [3.18286653, 6.59852554, 5.19721593, 2.43877   ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0002277
        self.a4a_true = -0.004348
        self.delta_phi_m3_true = -0.067025
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        elif sim2038_source:
            image_data = simulated_2038_image_data
        else:
            image_data = simple_image_data
        super(Mock2Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
