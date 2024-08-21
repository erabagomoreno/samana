import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_7_simple_dual import image_data as simple_image_data

class Mock7Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.35
        z_source = 1.4
        x_image = [-0.14992511,  0.96505004,  0.79256506, -0.53963   ]
        y_image = [ 1.08279281, -0.35217913,  0.64494747, -0.46184409]
        magnifications_true = [3.42699932, 4.06094774, 3.22541645, 1.07710349]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.001603
        self.a4a_true = 0.0145708
        self.delta_phi_m3_true = -0.443688
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock7Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


