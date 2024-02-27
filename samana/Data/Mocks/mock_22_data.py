import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_22_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_22_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock22Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.45
        z_source = 1.8
        x_image = [ 0.9202506 ,  0.05464368,  0.68946736, -0.70677944]
        y_image = [-0.69306623,  1.09370004,  0.74799816, -0.40755833]
        magnifications_true = [ 6.13159184, 17.90721766, 16.5958816 ,  3.33798114]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.013122
        self.a4a_true = -0.006050
        self.delta_phi_m3_true = -0.3052
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock22Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock22DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.45
        z_source = 1.8
        x_image = [-0.70207, 0.54739, -0.59454, 0.83076]
        y_image = [0.90964, -0.91275, -0.70352, 0.31262]
        magnifications_true = [6.04793, 7.84916, 5.92547, 5.14956]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.013122
        self.a4a_true = -0.006050
        self.delta_phi_m3_true = -0.3052
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock22DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
