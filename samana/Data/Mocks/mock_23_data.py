import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_23_cosmos import image_data as cosmos_image_data


class Mock23Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.45
        z_source = 1.68
        x_image = [-0.0607621 ,  0.22930363,  0.8515615 , -0.79221567]
        y_image = [-1.18887206,  0.92610005,  0.3725268 ,  0.35493974]
        magnifications_true = [2.4656894 , 6.25342743, 3.55684623, 2.63104752]
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
