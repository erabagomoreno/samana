import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_25_cosmos import image_data as cosmos_image_data


class Mock25Data(MockBase):

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
        super(Mock25Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
