import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_1_simple_dual import image_data as simple_image_data

class Mock19Data_dual(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.6
        z_source = 2.4
        x_image = [ 1.17795114, -0.75268413, -0.09844479, -0.50200288]
        y_image = [ 0.24500978, -0.61807482, -0.89309702,  0.63738784]
        magnifications_true = [2.85254683, 7.07738439, 5.66720369, 2.21696138]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00392
        self.a4a_true = 0.00810
        self.delta_phi_m3_true = -0.421461
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock19Data_dual, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)
