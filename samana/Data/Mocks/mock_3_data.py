import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_3_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_3_cosmos import image_data as cosmos_image_data

class Mock3Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.5
        z_source = 2.5
        x_image = [ 0.47825422, -0.59164391, -0.95062499,  0.65378873]
        y_image = [ 1.10369213, -0.84683562, -0.01019991, -0.56461223]
        magnifications_true = [ 3.91525799, 10.79471091,  5.23943925,  4.2675958 ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00514
        self.a4a_true = 0.01024
        self.delta_phi_m3_true = 0.053195
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            image_data = simple_image_data
        super(Mock3Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
