import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data


class Mock15Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.66
        z_source = 2.2
        x_image = [-0.91745675,  1.01379965,  0.18607431, -0.10975826]
        y_image = [-0.56844417, -0.12735224, -0.93809918,  0.78522661]
        magnifications_true = [4.27800327, 4.503553  , 4.71191955, 2.2988068 ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00680
        self.a4a_true = -0.00839
        self.delta_phi_m3_true = 0.36528
        self.delta_phi_m4_true = 0.0


        image_data = simple_image_data
        super(Mock15Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
