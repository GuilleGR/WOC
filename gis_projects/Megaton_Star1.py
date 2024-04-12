# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:24:34 2024

@author: Guillermo Rilova
"""
import os
import zipfile
import fiona
import geopandas as gpd
import numpy as np

#%% SITE 
from wind_stats.gwa_reader import get_gwc_data, get_weibull_parameters_sector
from wind_stats import WindDistribution 
#from py_wake.site import XRSite # IT DID NOT WORK
from py_wake.site import UniformWeibullSite

"FETCHING WIND CONDITIONS FOR SITE"

lat, lon =  32.928784, -101.729589
gwc_data = get_gwc_data(lat, lon)

roughness_lengths_distribution = [1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1]

A, k, f = get_weibull_parameters_sector(gwc_data, roughness_lengths_distribution, 115) # This comes from modifying the the returned variables from the function get_weibull_parameters
wd = np.linspace(0, 360, len(f), endpoint=False) # or gwc_data.sector
ti = .1
#%%
import pickle

with open(r"C:\Users\Guillermo Rilova\OneDrive - Greengo Energy\Documents\Wind\DEVELOPMENT\R&D\gis_projects\Megaton1_wind_conditions.pkl","wb") as fil:
    pickle.dump([A,k,f,wd,ti],fil)
fil.close()    
    
"CHANGE OF CONDAENVIRONMENT TO TEST"   

#%% LOADING THE DATA IN THE NEW ENVIRONMENT 
import pickle 

#from py_wake.site import XRSite # IT DID NOT WORK
from py_wake.site import UniformWeibullSite

fil = open(r"C:\Users\Guillermo Rilova\OneDrive - Greengo Energy\Documents\Wind\DEVELOPMENT\R&D\gis_projects\Megaton1_wind_conditions.pkl", 'rb')
obj = pickle.load(fil)
fil.close()


#%% DEVELOPABLE AREA

# Path to .kmz file 
kmz_file_path = r"C:\Users\Guillermo Rilova\OneDrive - Greengo Energy\Documents\Wind\DEVELOPMENT\R&D\gis_projects\MegatonStar1_BuildableArea.kmz"

# Specify the directory where we want to extract the KML file (same directory as the KMZ file)
extraction_dir = os.path.dirname(kmz_file_path)

# Open the KMZ file and extract its contents
with zipfile.ZipFile(kmz_file_path, "r") as kmz:
    kmz.extractall(extraction_dir)
    
# To make sure drivers are available    
fiona.drvsupport.supported_drivers['libkml'] = 'rw' 
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# Load the developable area
gdf = gpd.read_file(r"C:\Users\Guillermo Rilova\OneDrive - Greengo Energy\Documents\Wind\DEVELOPMENT\R&D\gis_projects\doc.kml", driver='libkml')
#Alternatively
#gdf2 = gpd.read_file(kmz_file_path)

# Plotting dev_area
gdf.geometry.plot()

#%% WIND TURBINES
from WOC_packages import catalogue as ctl
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

ls_tiki = ctl.list_wtg()

wtg_A = ctl.load_wtg('Vestas V136 4.5')
wtg_B = ctl.load_wtg('Vestas V172 7.2')


my_wt_A = WindTurbine(name='V136 4.5',
                    diameter=136,
                    hub_height=112,
                    powerCtFunction=PowerCtTabular(wtg_A.WS ,wtg_A.Power,'kW',wtg_A.CT))

my_wt_B = WindTurbine(name='V172 7.2',
                    diameter=172,
                    hub_height=114,
                    powerCtFunction=PowerCtTabular(wtg_A.WS ,wtg_A.Power,'kW',wtg_A.CT))

u = np.linspace(3, 27.5,num=50, endpoint=True)
ct = [0.894, 0.876, 0.856, 0.838, 0.825, 0.820, 0.821, 0.824, 0.825, 0.823, 0.812,  
      0.787, 0.750, 0.704, 0.653, 0.600, 0.545, 0.489, 0.436, 0.386, 0.342, 0.303, 
          0.269, 0.240, 0.216, 0.195, 0.176, 0.161, 0.147, 0.134, 0.123, 0.114, 
              0.105, 0.097, 0.090, 0.084, 0.078, 0.072, 0.067, 0.062, 0.058, 0.053, 
                  0.049,0.046, 0.043, 0.040, 0.037, 0.035, 0.033, 0]
    
power = [47, 126, 252, 415, 613, 848, 1128, 1457, 1840, 2281, 2775, 3312, 3868, 4421,
         4948, 5421, 5812, 6106, 6309, 6438, 6513, 6555, 6578, 6589, 6595, 6597, 6599,
         6599, 6600, 6600, 6599, 6597, 6592, 6581, 6562, 6531, 6486, 6423, 6342, 6246,
         6246, 6137, 6018, 5894, 5770, 5652, 5537, 5434, 5342, 5262, 0]

my_wt_C = WindTurbine(name='Custom_Siemens_6_6_155RD',
                    diameter=155,
                    hub_height=122.5,
                    powerCtFunction=PowerCtTabular(u,power,'kW',ct))



# Site with wind direction dependent weibull distributed wind speed
#uniform_weibull_site = XRSite(ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ti}, coords={'wd': wd})) It does not work

#uniform_weibull_site = UniformWeibullSite(f,                         # sector frequencies
##                          A,  # Weibull scale parameter
 #                         k,     # Weibull shape parameter
 #                         ti                                         # turbulence intensity, optional
 #                        )












