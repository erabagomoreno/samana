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



   

