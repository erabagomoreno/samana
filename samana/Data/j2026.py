import matplotlib.pyplot as plt
from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _J2026(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, z_lens):

        z_source = 2.23
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_J2026, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class J2026JWST(_J2026):

    def __init__(self, z_lens=0.5):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 0.10035525,  0.51610375,  0.26439393, -0.46879818])
        y_image = np.array([ 0.89672252, -0.31479607, -0.53410103, -0.14806814])

        image_position_uncertainties = [0.005] * 4 # 5 marcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J2026JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False, z_lens=z_lens)

class WFI2026_HST(_J2026):

    def __init__(self, z_lens=0.5):
        x = [0.187, 0.44, 0.023, -0.548]
        y = [-0.563, -0.348, 0.865, -0.179]
        m = [1.0, 0.75, 0.31, 0.28]
        image_position_uncertainties = [0.005] * 4  # 5 marcsec
        flux_uncertainties = [0.02, 0.02/0.75, 0.02/0.31, 0.01/0.28]
        super(WFI2026_HST, self).__init__(x, y, m, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=True, z_lens=z_lens)

#
# xc = np.array([-0.4985, 0.2364, 0.4897, 0.0725])
# yc = np.array([-0.2207, -0.6048, -0.3895, 0.8233])
# fc = np.array([0.288, 1.0, 0.893, 0.299])
# fc_ratios = fc
# lens = WFI2026_HST()
# flux_ratios = np.round(lens.magnifications/lens.magnifications[0],2)
# print(fc_ratios)
# colors = ['k', 'r','g','m']
# labels = ['A1', 'A2', 'B', 'C']
# for i in range(0, 4):
#
#     plt.scatter(xc[i], yc[i], color=colors[i],marker='+')
#     #plt.annotate(str(flux_ratios[i]),
#     #             xy=(lens.x_image[i], lens.y_image[i]+0.05),color=colors[i])
#     plt.annotate(str(fc_ratios[i]),
#                  xy=(xc[i], yc[i] - 0.05), color=colors[i])
#
# plt.show()
