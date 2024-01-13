import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_24_cosmos import image_data as cosmos_image_data


class Mock24Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.38
        z_source = 1.74
        x_image = [ 1.0021569 , -1.00226062, -0.53695543,  0.16859075]
        y_image = [ 0.5515738 , -0.24778595,  0.7462313 , -0.85669165]
        magnifications_true = [5.02526619, 6.19561855, 6.0815816 , 4.83956403]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00304
        self.a4a_true = -0.01228
        self.delta_phi_m3_true = 0.48172
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock24Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
