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
        Acoords = np.array([0, 0])
        Bcoords = np.array([0.138, 1.131])  #
        Ccoords = np.array([0.322, 1.531])
        Dcoords = np.array([1.288, 0.729])
        x = np.array([Acoords[0], Bcoords[0], Ccoords[0], Dcoords[0]])
        x_image = x - x.mean()
        y = np.array([Acoords[1], Bcoords[1], Ccoords[1], Dcoords[1]])
        y_image = y - y.mean()
        image_position_uncertainties = [0.005] * 4 # 5 marcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J0607JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)
'''
    def satellite_galaxy(self, sample=True):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """

        theta_E = 0.25
        Acoords = np.array([0,0])
        Ccoords = np.array([-0.79179357,-0.90458793])  #These names were reordered to be consistent with double dark matter vision
        Bcoords = np.array([-1.62141215,-0.59165656])
        Dcoords = np.array([-1.1289198, 0.15184604])
        x = np.array(Acoords[0],Bcoords[0],Ccoords[0],Dcoords[0])
        y = np.array(Acoords[1],Bcoords[1],Ccoords[1],Dcoords[1])
        g2_dra, g2_ddec = 0.477, -0.942 #arcsec, pos of g2 relative to B from double dark vision
        center_x = Bcoords[0] + g2_dra - x.mean()# these are positions from original coordinate system -0.307
        center_y = Bcoords[1] + g2_ddec - y.mean()# these are positions from original coordinate system -1.153

        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(self.zlens, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names
'''        
