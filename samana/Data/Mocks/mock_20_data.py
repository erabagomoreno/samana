import numpy as np
from samana.Data.Mocks.base import MockBase
#from samana.Data.ImageData.mock_20_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_20_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_20_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock20Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.35
        z_source = 2.1
        x_image = [ 0.02660204,  0.53298883,  0.94287462, -0.74628136]
        y_image = [-1.17162468,  0.8726903 ,  0.22653302,  0.22583915]
        magnifications_true = [3.28700437, 5.50374326, 4.30603636, 1.97959913]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004752
        self.a4a_true = 0.015022
        self.delta_phi_m3_true = 0.0922903
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock20Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock20DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.35
        z_source = 2.1

        x_image = [-0.04257, 0.49181, 0.95338, -0.76857]
        y_image = [-1.17621, 0.91002, 0.16629, 0.2458]
        magnifications_true = [3.43403, 5.39403, 3.83164, 2.35667]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004752
        self.a4a_true = 0.015022
        self.delta_phi_m3_true = 0.0922903
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock20DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
