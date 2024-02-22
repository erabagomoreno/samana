from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _J0607(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        z_lens = 0.5 #fiducial
        z_source = 1.305
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_J0607, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class J0607JWST(_J0607):

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
        #Bcoords = np.array([0.138, 1.131])  #
        #Ccoords = np.array([0.322, 1.531])
        #Dcoords = np.array([1.288, 0.729])
        #x = np.array([Acoords[0], Bcoords[0], Ccoords[0], Dcoords[0]])
        #x_image = x - x.mean()
        #y = np.array([Acoords[1], Bcoords[1], Ccoords[1], Dcoords[1]])
        #y_image = y - y.mean()
        x_image=  np.array([-0.67058978, -0.53238823, -0.34850586,  0.61742581])
        y_image=  np.array([-0.7192856,   0.41213327,  0.8118,      0.00952086])
        image_position_uncertainties = [0.005] * 4 # 5 marcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J0607JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)
