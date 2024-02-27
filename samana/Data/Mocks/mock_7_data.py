import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_7_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_7_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_7_cosmos_wdm import image_data as cosmos_image_data_wdm


class Mock7Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.35
        z_source = 1.4
        x_image = [-0.16234706,  0.98575445,  0.80004301, -0.55537481]
        y_image = [ 1.0989313 , -0.36710543,  0.65941259, -0.47587784]
        magnifications_true = [3.41113138, 4.05698314, 3.19503437, 1.14192612]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.001603
        self.a4a_true = 0.0145708
        self.delta_phi_m3_true = -0.443688
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            image_data = simple_image_data
        super(Mock7Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock7DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.35
        z_source = 1.4
        x_image = [-0.15145, 0.99289, 0.83214, -0.53633]
        y_image = [1.10992, -0.3208, 0.62988, -0.47178]
        magnifications_true = [3.26901, 4.50396, 3.52181, 1.08651]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.001603
        self.a4a_true = 0.0145708
        self.delta_phi_m3_true = -0.443688
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock7DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
