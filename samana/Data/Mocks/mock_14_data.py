import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_14_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_14_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_14_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock14Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.3
        z_source = 1.8
        x_image = [-1.14366076,  0.94735354, -0.11859553,  0.24369212]
        y_image = [ 0.01717449,  0.60062471,  0.96083746, -0.80040258]
        magnifications_true = [4.78546426, 5.46885119, 6.10470246, 2.79702338]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00231
        self.a4a_true = 0.0034161
        self.delta_phi_m3_true = 0.014601
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            image_data = simple_image_data
        super(Mock14Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock14DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.3
        z_source = 1.8

        x_image = [-1.13387, 0.94269, -0.12449, 0.25924]
        y_image = [-0.03709, 0.59296, 0.94759, -0.79465]
        magnifications_true = [4.84949, 5.98865, 6.27305, 2.85838]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00231
        self.a4a_true = 0.0034161
        self.delta_phi_m3_true = 0.014601
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock14DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
