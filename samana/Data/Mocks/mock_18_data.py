import numpy as np
from samana.Data.Mocks.base import MockBase
#from samana.Data.ImageData.mock_18_simple import image_data as simple_image_data
from samana.Data.ImageData.MockImageData.mock_18_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.MockImageData.mock_18_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock18Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False):

        z_lens = 0.52
        z_source = 1.82
        x_image = [-0.93950917,  0.94490824,  0.54693728, -0.38600058]
        y_image = [ 0.5649149 , -0.48974962,  0.75141613, -0.84185525]
        magnifications_true = [ 6.93924524, 10.23747022,  6.7625307 ,  5.82411634]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0052722
        self.a4a_true = 0.000151
        self.delta_phi_m3_true = 0.1574715
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        else:
            raise Exception('not yet implemented')
            # image_data = simple_image_data
        super(Mock18Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock18DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.52
        z_source = 1.82

        x_image = [-0.93539, 0.93539, 0.53045, -0.37893]
        y_image = [0.53141, -0.48613, 0.7563, -0.83627]
        magnifications_true = [6.62491, 8.34924, 6.93076, 6.7203]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0052722
        self.a4a_true = 0.000151
        self.delta_phi_m3_true = 0.1574715
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock18DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                image_data, super_sample_factor)
