import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_5_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_5_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_5_2038 import image_data as simulated_2038_image_data
from samana.Data.ImageData.mock_5_cosmos_wdm import image_data as cosmos_image_data_wdm


class Mock5Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False):

        z_lens = 0.4
        z_source = 1.6
        x_image = [-1.04814384,  0.99262214,  0.50969389, -0.30372116]
        y_image = [ 0.47660129, -0.42046054,  0.73688046, -0.8312197 ]
        magnifications_true = [3.06880609, 3.54699025, 2.34253429, 2.36118715]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00146
        self.a4a_true = 0.00371
        self.delta_phi_m3_true = -0.2911280
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        elif sim2038_source:
            image_data = simulated_2038_image_data
        else:
            image_data = simple_image_data
        super(Mock5Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock5DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.4
        z_source = 1.6
        x_image = [-1.07495, 0.96647, -0.24352, 0.53939]
        y_image = [0.44938, -0.43183, -0.85204, 0.71305]
        magnifications_true = [2.84354, 3.80529, 2.55344, 2.28153]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00146
        self.a4a_true = 0.00371
        self.delta_phi_m3_true = -0.2911280
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock5DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
