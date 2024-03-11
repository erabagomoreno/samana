import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData.mock_15_simple import image_data as simple_image_data
from samana.Data.ImageData.MockImageData.mock_15_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.MockImageData.mock_15_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock15Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.66
        z_source = 2.2
        x_image = [-0.96089407,  1.04418626,  0.21704251, -0.08375568]
        y_image = [-0.61282176, -0.20067121, -0.9824735 ,  0.8015401 ]
        magnifications_true = [4.42169713, 4.67852868, 5.42764821, 2.3811]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00680
        self.a4a_true = -0.00839
        self.delta_phi_m3_true = 0.36528
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            image_data = simple_image_data
        super(Mock15Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock15DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.66
        z_source = 2.2

        x_image = [-0.93826, 1.05479, 0.15999, -0.12455]
        y_image = [-0.60334, -0.11294, -0.98545, 0.81685]
        magnifications_true = [5.20366, 5.20219, 5.27341, 2.60167]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00680
        self.a4a_true = -0.00839
        self.delta_phi_m3_true = 0.36528
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock15DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
