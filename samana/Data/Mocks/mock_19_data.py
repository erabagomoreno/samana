import numpy as np
from samana.Data.Mocks.base import MockBase
#from samana.Data.ImageData.mock_19_simple import image_data as simple_image_data
from samana.Data.ImageData.mock_19_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.mock_19_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock19Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.6
        z_source = 2.4
        x_image = [ 1.19445914, -0.85270481,  0.06874894, -0.4683292 ]
        y_image = [ 0.24383119, -0.58781733, -0.91869502,  0.69210194]
        magnifications_true = [2.88538113, 6.00061622, 4.08394975, 2.19519151]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00392
        self.a4a_true = 0.00810
        self.delta_phi_m3_true = -0.421461
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            # image_data = simple_image_data
        super(Mock19Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock19DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.6
        z_source = 2.4

        x_image = [1.1732, -0.85338, 0.00603, -0.45346]
        y_image = [0.21485, -0.56293, -0.92511, 0.69242]
        magnifications_true = [3.28719, 6.41986, 4.88605, 2.17879]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00392
        self.a4a_true = 0.00810
        self.delta_phi_m3_true = -0.421461
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock19DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
