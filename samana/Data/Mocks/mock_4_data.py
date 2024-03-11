import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData.mock_4_simple import image_data as simple_image_data
from samana.Data.ImageData.MockImageData.mock_4_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.MockImageData.mock_4_2038 import image_data as simulated_2038_image_data
from samana.Data.ImageData.MockImageData.mock_4_cosmos_wdm import image_data as cosmos_image_data_wdm
from samana.Data.ImageData.MockImageData.mock_4_cosmos_highSNR import image_data as cosmos_image_data_highSNR


class Mock4Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.5
        z_source = 1.5
        x_image = [ 0.26197365,  0.37192752,  0.99587974, -0.89833527]
        y_image = [-0.98968167,  0.99370493,  0.21072101, -0.01553163]
        magnifications_true = [5.1207985 , 5.22847917, 5.41861705, 2.92272106]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00225
        self.a4a_true = 0.000450
        self.delta_phi_m3_true = 0.4890
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        elif sim2038_source:
            image_data = simulated_2038_image_data
        else:
            image_data = simple_image_data
        super(Mock4Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock4DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.5
        z_source = 1.5
        x_image = [0.3164, 0.411, 0.98824, -0.88536]
        y_image = [-0.97356, 0.96816, 0.24091, -0.04522]
        magnifications_true = [5.42615, 5.24868, 5.74547, 2.67319]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00225
        self.a4a_true = 0.000450
        self.delta_phi_m3_true = 0.4890
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock4DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)

class Mock4DataHighSNR(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.5
        z_source = 1.5
        x_image = [0.28055, 0.34591, 0.95996, -0.87343]
        y_image = [-0.96895, 0.96808, 0.2571, -0.03481]
        magnifications_true = [4.71319, 5.02965, 5.06041, 2.67437]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00225
        self.a4a_true = 0.000450
        self.delta_phi_m3_true = 0.4890
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_highSNR
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock4DataHighSNR, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
