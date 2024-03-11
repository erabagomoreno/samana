import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData.mock_23_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.MockImageData.mock_23_cosmos_wdm import image_data as cosmos_image_data_wdm
from samana.Data.ImageData.MockImageData.mock_23_cosmos_highSNR import image_data as cosmos_image_data_highSNR


class Mock23Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.45
        z_source = 1.68
        x_image = [-0.0607621 ,  0.22930363,  0.8515615 , -0.79221567]
        y_image = [-1.18887206,  0.92610005,  0.3725268 ,  0.35493974]
        magnifications_true = [2.4656894 , 6.25342743, 3.55684623, 2.63104752]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.0046
        self.a4a_true = -0.00723
        self.delta_phi_m3_true = 0.01811
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock23Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock23DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.45
        z_source = 1.68
        x_image = [-0.02107, 0.30404, 0.8528, -0.78146]
        y_image = [-1.19311, 0.91493, 0.38652, 0.34054]
        magnifications_true = [2.3654, 5.60175, 3.80994, 2.27911]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.0046
        self.a4a_true = -0.00723
        self.delta_phi_m3_true = 0.01811
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock23DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)

class Mock23DataHighSNR(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.45
        z_source = 1.68
        x_image = [-0.02219, 0.23754, 0.82754, -0.7676]
        y_image = [-1.16441, 0.90357, 0.38062, 0.35362]
        magnifications_true = [2.30308, 5.45543, 3.48139, 2.46878]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.0046
        self.a4a_true = -0.00723
        self.delta_phi_m3_true = 0.01811
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_highSNR
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock23DataHighSNR, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
