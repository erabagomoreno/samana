#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:13:17 2024

@author: ryankeeley
"""

import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.data_base import ImagingDataBase
from samana.Data.ImageData.no_imaging_data import image_data as no_imaging_data




class J1606Data(MockBase):#Replace this with ImagingDataBase eventually?
    
    def __init__(self, image_position_uncertainties, flux_uncertainties, magnifications,
                 uncertainty_in_fluxes, supersample_factor):
        
        z_lens = 0.5
        z_source = 1.69
        
        Acoords = np.array([0,0])
        Ccoords = np.array([-0.79179357,-0.90458793])  #These names were reordered to be consistent with double dark matter vision
        Bcoords = np.array([-1.62141215,-0.59165656])
        Dcoords = np.array([-1.1289198, 0.15184604])
        x = np.array(Acoords[0],Bcoords[0],Ccoords[0],Dcoords[0])
        x_image = x - x.mean()
        y = np.array(Acoords[1],Bcoords[1],Ccoords[1],Dcoords[1])
        y_image = y - y.mean()
        magnifications = [1.0, 1.0, 1.0, 1.0]  # note to Ryan, still need to replace these with results from SED fitting
        flux_ratio_uncertainties = None
        astrometric_uncertainties = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]# this line is hard coded in the MockBase class
        image_data = no_imaging_data
        super(J1606, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                        image_data, supersample_factor)



   




#what to do with the satellite galaxy?

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