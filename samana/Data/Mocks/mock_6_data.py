import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData.mock_6_simple import image_data as simple_image_data
from samana.Data.ImageData.MockImageData.mock_6_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.MockImageData.mock_6_2038 import image_data as simulated_2038_image_data
from samana.Data.ImageData.MockImageData.mock_6_cosmos_wdm import image_data as cosmos_image_data_wdm
from samana.Data.ImageData.MockImageData.mock_6_cosmos_highSNR import image_data as cosmos_image_data_highSNR


class Mock6Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.55
        z_source = 2.6
        x_image = [ 0.62623824, -0.23244766,  0.7292022 , -0.80853096]
        y_image = [-0.91953787,  0.97866734,  0.62435675, -0.30273136]
        magnifications_true = [4.07148687, 4.354839, 3.40278675, 2.3364734 ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00502
        self.a4a_true = -0.005256
        self.delta_phi_m3_true = 0.411402
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        elif sim2038_source:
            image_data = simulated_2038_image_data
        else:
            image_data = simple_image_data
        super(Mock6Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock6DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.55
        z_source = 2.6
        x_image = [0.62986, -0.08794, 0.62958, -0.79926]
        y_image = [-0.93614, 0.97534, 0.72808, -0.19462]
        magnifications_true = [2.96931, 5.99645, 4.38028, 2.34592]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = 0.00502
        self.a4a_true = -0.005256
        self.delta_phi_m3_true = 0.411402
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock6DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)

class Mock6DataHighSNR(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.55
        z_source = 2.6

        x_image = [0.6132, -0.16838, 0.67837, -0.76361]
        y_image = [-0.88214, 0.95341, 0.63409, -0.25116]
        magnifications_true = [3.20902, 4.37669, 3.30329, 2.03231]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = 0.00502
        self.a4a_true = -0.005256
        self.delta_phi_m3_true = 0.411402
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_highSNR
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock6DataHighSNR, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
