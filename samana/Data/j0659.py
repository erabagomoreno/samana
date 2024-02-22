from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _J0659(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        z_lens = 0.77
        z_source = 3.1
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_J0659, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class J0659JWST(_J0659):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        #Acoords = np.array([0, 0])
        #Ccoords = np.array(
        #    [-0.79179357, -0.90458793])  # These names were reordered to be consistent with double dark matter vision
        #Bcoords = np.array([-1.62141215, -0.59165656])
        #Dcoords = np.array([-1.1289198, 0.15184604])
        #x = np.array([Acoords[0], Bcoords[0], Ccoords[0], Dcoords[0]])
        #x_image = x - x.mean()
        #y = np.array([Acoords[1], Bcoords[1], Ccoords[1], Dcoords[1]])
        #y_image = y - y.mean()
        x_image = np.array([ 1.92617515, -2.73882485,  0.94717515,  2.01017515])
        y_image = np.array([-0.90865046,-1.24365046,  1.98334954,  0.99434954])
        #x_image = np.array([ 0.86042523, -0.76106166, 0.06865239, -0.26849227])
        #y_image = np.array([ 0.35832518, -0.23329177, -0.54623962, 0.51033051])
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J0659JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)
