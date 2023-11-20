import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.mock_2_simple import image_data

class Mock2Data(MockBase):

    def __init__(self):

        z_lens = 0.6
        z_source = 1.9
        x_image = [ 1.01542165, -0.90180182, -0.68849556,  0.17478882]
        y_image = [ 0.69443497, -0.42329349,  0.59061389, -0.8711085 ]
        magnifications_true = [3.47005252, 6.09434583, 4.45503079, 2.81007421]
        magnification_measurement_errors = [-0.04338515, -0.01028729, -0.28550458,  0.13827848]
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.005] * 4
        flux_uncertainties = [0.03] * 4
        super(Mock2Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_uncertainties, image_data)
