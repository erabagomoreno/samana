from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _J1537(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        #z_lens = 0.592
        z_lens = 0.6
        #z_source = 1.721
        z_source = 1.7
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_J1537, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class J1537JWST(_J1537):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        #0,-1.993,-2.848,-0.750
        #0,-0.329,1.644,1.763
        #Acoords = np.array([0.0, 0.0])
        #Bcoords = np.array([-1.993, -0.329])
        #Ccoords = np.array([-2.848, 1.644]) 
        #Dcoords = np.array([-0.750, 1.763])
        #x = np.array([Acoords[0], Bcoords[0], Ccoords[0], Dcoords[0]])
        #x_image = x - x.mean()
        #y = np.array([Acoords[1], Bcoords[1], Ccoords[1], Dcoords[1]])
        #y_image = y - y.mean()
        x_image = np.array( [ 1.42556606, -0.56743394, -1.42243394,  0.67556606])
        y_image = np.array( [-0.71662386, -1.04562386,  0.92737614,  1.04637614])
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J1537JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                                uncertainty_in_fluxes=False)
