import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_25_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_25_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock25DataOld(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.52
        z_source = 2.06
        x_image = [-1.11375629,  0.67228686,  0.0361663 ,  0.73541128]
        y_image = [-0.55479773,  0.67801258,  0.90018184, -0.44997673]
        magnifications_true = [2.68351655, 8.48277454, 5.70063327, 3.08530137]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.008121
        self.a4a_true = 0.006042
        self.delta_phi_m3_true =  0.387593
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock25DataOld, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

class Mock25Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.52
        z_source = 1.55
        x_image = [-1.1647733 ,  0.32034898, -0.3577157 ,  0.4567963 ]
        y_image = [-0.04894637,  1.03652736,  0.98925922, -0.61337704]
        magnifications_true = [4.08194276, 6.85643424, 8.05628072, 1.60356098]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.008121
        self.a4a_true = 0.006042
        self.delta_phi_m3_true = 0.387593
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock25Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock25DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.52
        z_source = 1.55
        x_image = [-1.15086, 0.34304, -0.41321, 0.44517]
        y_image = [0.00574, 1.03228, 0.97384, -0.63128]
        magnifications_true = [4.78229, 7.11359, 7.05763, 1.66948]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.008121
        self.a4a_true = 0.006042
        self.delta_phi_m3_true = 0.387593
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock25DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
