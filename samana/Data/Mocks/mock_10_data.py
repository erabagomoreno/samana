import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_10_simple import image_data

class Mock10Data(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.45
        z_source = 3.1
        x_image = [-0.83974435,  0.70954169, -0.53806097,  0.69010008]
        y_image = [ 0.7673927 , -0.7646517 , -0.73906562,  0.55370045]
        magnifications_true = [3.8218467 , 4.46562561, 3.36502876, 2.88481672]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0001888
        self.a4a_true = -0.0013544
        self.delta_phi_m3_true = 0.28412
        self.delta_phi_m4_true = 0.0

        super(Mock10Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
