import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData_dual.mock_3_simple_dual import image_data as simple_image_data

class Mock3Data_dual_wdm(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False, cosmos_source_psf3=False):

        z_lens = 0.5
        z_source = 2.5
        x_image = [ 0.46890765, -0.56898087, -0.9097677 ,  0.58257924]
        y_image = [ 1.06020183, -0.81342863, -0.01987317, -0.6112157 ]
        magnifications_true = [3.94333007, 7.8155774 , 5.06759841, 4.33422744]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.00514
        self.a4a_true = 0.01024
        self.delta_phi_m3_true = 0.053195
        self.delta_phi_m4_true = 0.0

        image_data = simple_image_data
        super(Mock3Data_dual_wdm, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)




