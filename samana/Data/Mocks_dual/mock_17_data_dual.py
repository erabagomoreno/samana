import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data

class Mock17Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.45
        z_source = 1.6
        x_image = [ 1.08331275,  0.2730449 ,  0.91284344, -0.56928877]
        y_image = [-0.38839497,  1.07250491,  0.57473611, -0.29387153]
        magnifications_true = [5.64892071, 6.1804664 , 7.59023925, 1.16342712]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00688
        self.a4a_true = 0.001364
        self.delta_phi_m3_true = -0.21502
        self.delta_phi_m4_true = 0.0


        image_data = simple_image_data
        super(Mock17Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
