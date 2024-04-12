# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:42:05 2024

@author: Guillermo Rilova
"""

import rioxarray as rxr

FILEPATH = r'C:\Users\Guillermo Rilova\OneDrive - Greengo Energy\Documents\Wind\DEVELOPMENT\R&D\WindClimate\ESP_wind-speed_100m.tif'

dataarray = rxr.open_rasterio(FILEPATH) # Only fetches wind speed

#%%

from wind_stats import get_gwc_data
from wind_stats import WindDistribution 
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter 
from wind_stats.gwa_reader import get_gwc_data, get_weibull_parameters



lat, lon = 41.061, -6.1445

gwc_data = get_gwc_data(lat, lon)

roughness_lengths_distribution = [1, 1, 1, 0.9, 0.7, 0.7, 1, 1 , 1, 1, 1, 1]

A, k, frequencies = get_weibull_parameters(gwc_data, 1.5, 135)


wind_distribution = WindDistribution.from_gwc(gwc_data, roughness_lengths_distribution, 150)

#%% 
from wind_stats import Site

Site.from_gwc(lat, lon, roughness_lengths_distribution, 150)



