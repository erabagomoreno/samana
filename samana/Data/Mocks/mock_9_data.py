import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_9_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_9_cosmos_wdm import image_data as cosmos_image_data_wdm


class Mock9Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.6
        z_source = 1.79
        x_image = [-0.69393437,  0.83384171,  0.24655443, -0.05964653]
        y_image = [ 0.81722545,  0.73098528,  1.07274727, -0.76767204]
        magnifications_true = [ 9.12504919, 10.74884219, 14.36750772,  1.94103826]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.005
        self.a4a_true = -0.01
        self.delta_phi_m3_true = -0.5127
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock9Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock9DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.6
        z_source = 1.79
        x_image = [-0.68183, 0.79249, 0.1812, -0.07165]
        y_image = [0.82811, 0.78187, 1.08895, -0.77039]
        magnifications_true = [8.97922, 12.06119, 14.20183, 1.98229]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.005
        self.a4a_true = -0.01
        self.delta_phi_m3_true = -0.5127
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock9DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
