import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_3_simple import image_data

class Mock3Data(MockBase):

    def __init__(self):

        z_lens = 0.7
        z_source = 3.2
        x_image = [0.70200517, -0.23995158,  0.6139985, -0.88422848]
        y_image = [1.0200306, -0.99535205, -0.69992801,  0.10487893]
        magnifications_true = [3.81096342, 7.9771231, 7.1110618, 3.73881453]
        magnification_measurement_errors = [0.20449193,  0.10446278,  0.02058598, -0.20901761]
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.005] * 4
        flux_uncertainties = [0.03] * 4
        super(Mock3Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_uncertainties, image_data)
