import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_23_cosmos import image_data as cosmos_image_data


class Mock23Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.46
        z_source = 2.7
        x_image = [ 0.55194149,  0.00814457,  0.86172201, -0.90168277]
        y_image = [ 0.95585811, -1.04971912, -0.41817934,  0.08966123]
        magnifications_true = [5.28664498, 7.44644454, 6.86279627, 3.86151631]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.0046
        self.a4a_true = -0.00723
        self.delta_phi_m3_true = 0.01811
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock23Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
