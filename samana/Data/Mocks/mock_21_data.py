import numpy as np
from samana.Data.Mocks.base import MockBase
#from samana.Data.ImageData.mock_4v2 import image_data as simple_image_data
from samana.Data.ImageData.mock_21_cosmos import image_data as cosmos_image_data


class Mock21Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.76
        z_source = 2.9
        x_image = [ 0.26010583,  0.35122995,  0.91101153, -0.90076471]
        y_image = [ 1.06865352, -0.92303886, -0.31716535, -0.22975958]
        magnifications_true = [4.52685823, 8.35380825, 8.49719646, 4.01043563]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00225
        self.a4a_true = 0.000450
        self.delta_phi_m3_true = 0.4890725
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock21Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
