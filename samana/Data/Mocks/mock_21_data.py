import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData.mock_21_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.MockImageData.mock_21_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock21Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.6
        z_source = 2.45
        x_image = [ 0.79902184, -0.53863238,  0.4233966 , -0.83534616]
        y_image = [ 0.82498761, -0.89673085, -0.85079082,  0.34237227]
        magnifications_true = [ 5.51402665, 10.90472163,  8.91978704,  5.88926075]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00507
        self.a4a_true = -0.009693
        self.delta_phi_m3_true = -0.47257
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock21Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock21DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.6
        z_source = 2.45
        x_image = [0.75082, -0.45755, 0.446, -0.8538]
        y_image = [0.87049, -0.9151, -0.83913, 0.26534]
        magnifications_true = [5.97985, 13.12586, 11.35459, 5.9454]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.00507
        self.a4a_true = -0.009693
        self.delta_phi_m3_true = -0.47257
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock21DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
