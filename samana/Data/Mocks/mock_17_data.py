import numpy as np
from samana.Data.Mocks.base import MockBase
#from samana.Data.ImageData.mock_1 _simple import image_data as simple_image_data
from samana.Data.ImageData.mock_17_cosmos import image_data as cosmos_image_data


class Mock17Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.45
        z_source = 1.6
        x_image = [ 1.09522555,  0.24619197,  0.90316384, -0.60910382]
        y_image = [-0.44010706,  1.11564998,  0.64834563, -0.29683906]
        magnifications_true = [4.88635084, 7.40289424, 8.03114211, 1.29929075]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00688
        self.a4a_true = 0.001364
        self.delta_phi_m3_true = -0.21502
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            #image_data = simple_image_data
        super(Mock17Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
