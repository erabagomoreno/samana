import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_23_cosmos import image_data as cosmos_image_data


class Mock23Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.46
        z_source = 2.7
        x_image = [-1.03172646,  0.61729877, -0.34614795,  0.38593992]
        y_image = [-0.00899676,  0.86855948,  0.94332842, -0.86147284]
        magnifications_true = [ 9.62093838, 10.53799588, 12.08146717,  6.54396696]
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
