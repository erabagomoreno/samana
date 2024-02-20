import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_11_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_11_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_11_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock11Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.6
        z_source = 1.9
        x_image = [ 0.99983001, -0.39159643,  0.71706307, -0.53202555]
        y_image = [-0.49044378,  1.07314247,  0.70653125, -0.54045528]
        magnifications_true = [4.00762705, 4.15880995, 4.50209536, 1.64640534]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00014642
        self.a4a_true = -0.003825
        self.delta_phi_m3_true = -0.3348207
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            image_data = simple_image_data
        super(Mock11Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

class Mock11DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.6
        z_source = 1.9
        x_image = [1.02087, -0.39666, 0.68465, -0.54517]
        y_image = [-0.46983, 1.08934, 0.76597, -0.53509]
        magnifications_true = [4.21291, 4.45266, 4.48534, 1.75476]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00014642
        self.a4a_true = -0.003825
        self.delta_phi_m3_true = -0.3348207
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock11DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
