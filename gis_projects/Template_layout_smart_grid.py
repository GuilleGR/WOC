# -*- coding: utf-8 -*-
"""
Created on Tues March 19 2024

@author: Guillermo Rilova
"""

#!/usr/bin/env python
# coding: utf-8

# ### Create a directory to store GIS project and get libraries

# In[1]:


import os
import sys
import pandas as pd
import numpy as np

import geopandas as gpd
import fiona
#import rasterio


# make directory called 'gis_projects' in c drive if not exists

if not os.path.exists('C:\\Users\\Guillermo Rilova\\OneDrive - Greengo Energy\\Documents\\Wind\DEVELOPMENT\R&D\\gis_projects'):
    os.makedirs('C:\\Users\\Guillermo Rilova\\OneDrive - Greengo Energy\\Documents\\Wind\DEVELOPMENT\R&D\\gis_projects')
    
    
    
#%%

from arcgis.features import FeatureLayer
from arcgis.gis import GIS
from arcgis.features import FeatureLayerCollection
from arcgis.geometry import Geometry
from arcgis.features import Feature, FeatureSet

import geopandas as gpd

class ArcGISHandler:
    def __init__(self):
        # Read the configuration from a JSON file
        # with open(r'D:/Python/Production/Config_files/config.json', 'r') as f:
        #     config = json.load(f)

        # # Extract GIS configuration and initialize GIS object
        # gis_config = config['online_gis']
        # self.gis = GIS(gis_config['url'], gis_config['username'], gis_config['password'])

        # ! Recommend putting details into JSON as above - switch out with Thomas' details for now (I already share mine with Mara - there is a limit to how many people can share a single account)
        self.gis = GIS('https://greengo-eu.maps.arcgis.com', 'GGE_RT', 'ARcgispro13*')  # Ray's credentials

    def fetch_data_gdf(self, url: str) -> gpd.GeoDataFrame:
        try:
            # Create a FeatureLayer object
            feature_layer = FeatureLayer(url, self.gis)
            
            # Query all features in the layer with the query() method
            feature_set = feature_layer.query(where='1=1')
            
            # Convert the features to a Spatially Enabled DataFrame
            sdf = feature_set.sdf
            
            # Convert the Spatially Enabled DataFrame to a GeoDataFrame
            gdf = gpd.GeoDataFrame(sdf, geometry=sdf['SHAPE'])
            
            return gdf
        except Exception as e:
            print(f"An error occurred while fetching data from ArcGIS: {e}")
            return None
        
#%% Load shape file as polygon 
#gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

#dev_area = gpd.read_file(r"C:\Users\Guillermo Rilova\OneDrive - Greengo Energy\Documents\Wind\DEVELOPMENT\R&D\gis_projects\Test_dev_area_sugar.kml", driver='KML')



#%%

from catalogue import load_wtg_P


r'C:\Users\Guillermo Rilova\OneDrive - Greengo Energy\Documents\Wind\DEVELOPMENT\R&D\gis_projects\WOC\Wind_turbines_catalogue.xlsx', 'Vestas V172-7.2-PC'

#%% get some data


### get DK projects
project_url = 'https://services-eu1.arcgis.com/Iu9KJRhfAInbTSB3/arcgis/rest/services/Denmark_Projects_1f1dd/FeatureServer/0'
arcgis_handler = ArcGISHandler()
project_dk_gdf = arcgis_handler.fetch_data_gdf(project_url)



windfarms_url = 'https://services-eu1.arcgis.com/Iu9KJRhfAInbTSB3/arcgis/rest/services/Windmills_2021/FeatureServer/0'
arcgis_handler = ArcGISHandler()
windfarms_dk_gdf = arcgis_handler.fetch_data_gdf(windfarms_url)

project_url = 'https://services-eu1.arcgis.com/Iu9KJRhfAInbTSB3/arcgis/rest/services/Neighbour_status/FeatureServer/0'
arcgis_handler = ArcGISHandler()
Neighbour_status = arcgis_handler.fetch_data_gdf(project_url)


# read a shapefile
# shapefile = 'XXXXXXXX\\DK.shp'
# shapefile_gdf = gpd.read_file(shapefile)

# print number of windfarms in DK
print(f"Number of windfarms: {len(windfarms_dk_gdf)}")
print(windfarms_dk_gdf.columns)
print(f"Number of projects: {len(project_dk_gdf)}")
print(project_dk_gdf.columns)


windfarms_dk_gdf.plot()

windfarms_dk_gdf.head()
#%% Filtering Ulsted project


project_Ulsted = project_dk_gdf[project_dk_gdf['Project_Name'] == 'M147 - Ulsted Kær']
#existing_Ulsted = windfarms_dk_gdf # Cannot filter this url per project
#neighbours = Neighbour_status # Not consistent naming for the layers[Neighbour_status['ProjectArea'] == 'M147 - Ulsted Kær'] 

#%% Calculate the bounds of the project area and expand by 1000m
minx, miny, maxx, maxy = project_Ulsted.total_bounds

buffer = 4000  # 4000 meters buffer
bounds = [minx - buffer, miny - buffer, maxx + buffer, maxy + buffer]

#%% filter the Neighbours and the wind turbines withing the area of influence

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def def_polygon_bound(bounds):
    polygon_obj = Polygon([(bounds[0],bounds[1]),(bounds[0],bounds[3]),(bounds[2],bounds[3]),(bounds[2],bounds[1])]) # (minx,miny), (minx,maxy), (maxx,maxy), (maxx,miny)
    return polygon_obj

polygon = def_polygon_bound(bounds) # Defining polygon bounds for the project

neighbours = Neighbour_status[(polygon.contains(Neighbour_status.geometry))]
existing_Ulsted  = windfarms_dk_gdf[(polygon.contains(windfarms_dk_gdf.geometry))]

#% add buffer over neighbours
neighbours_buffer = neighbours.copy()
neighbours_buffer['geometry'] = neighbours_buffer.geometry.buffer(600)
#neighbours_buffer.plot(ax=ax, color='yellow', alpha=0.2, edgecolor='k')
buffers_dissolved = neighbours_buffer.dissolve()

#%% Add buffer to existing wtg for testing purposes
existing_buffer = existing_Ulsted.copy()
existing_buffer['geometry'] = existing_buffer.geometry.buffer(600)

existing_buffers_dissolved = existing_buffer.dissolve()



#%%

import matplotlib.pyplot as plt


# Create a plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the project area
project_Ulsted.plot(ax=ax, color='blue', edgecolor='k')

# Plot the windfarms - ORIGINAL WINDFARMS
existing_Ulsted.plot(ax=ax, color='green', edgecolor='k')

# Plot the neighbours 
neighbours.plot(ax=ax, color='red', edgecolor='k')

# Set the plot limits to the calculated bounds
#ax.set_xlim([bounds[0], bounds[2]])
#ax.set_ylim([bounds[1], bounds[3]])
#ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:25832')

buffers_dissolved.plot(ax=ax, color='pink', alpha=0.4, edgecolor='k')
existing_buffers_dissolved.plot(ax=ax, color='green', alpha=0.5, edgecolor='k')


# Optional: Add grid, labels, title, etc.
ax.grid(True)
ax.set_title('Project Area and Windfarms')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')


# Show the plot

plt.show()

#%% Difference 

dev_area = project_Ulsted.overlay(existing_buffer, how='difference')


#%%
import matplotlib.pyplot as plt
 
fig2, ax2 = plt.subplots(figsize=(10, 10))

dev_area.plot(ax=ax2, color='blue', edgecolor='k')

#existing_buffers_dissolved.plot(ax=ax2, color='green', alpha=0.3, edgecolor='k')

# Optional: Add grid, labels, title, etc.
ax2.grid(True)
ax2.set_title('Project Area and Windfarms')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')


# Show the plot

plt.show()

#%% Climate
from py_wake.site import UniformWeibullSite
#specifying the necessary parameters for the UniformWeibullSite object
site = UniformWeibullSite(p_wd = [.20,.25,.35,.25],                         # sector frequencies
                          a = [9.176929,  9.782334,  9.531809,  9.909545],  # Weibull scale parameter
                          k = [2.392578, 2.447266, 2.412109, 2.591797],     # Weibull shape parameter
                          ti = 0.1                                          # turbulence intensity, optional
                         )
#%% Generic wtg
import numpy as np
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


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

my_wt = WindTurbine(name='Custom_Siemens_6_6_155RD',
                    diameter=155,
                    hub_height=122.5,
                    powerCtFunction=PowerCtTabular(u,power,'kW',ct))


 
#%%

def initial_positions(DEV_AREA,n_wt):
    wt_x =[]
    wt_y =[]
    count=1
    iterat= 0
    while count< n_wt+1:
       WT_x, WT_y = np.random.uniform(x_min, x_max, 1), np.random.uniform(y_min, y_max, 1)
       iterat+=1
       print(iterat)
       if (DEV_AREA.contains(Point([WT_x,WT_y])))> 0:
           print(DEV_AREA.contains(Point([WT_x,WT_y])),WT_x,WT_y)
           wt_x.append(WT_x), wt_y.append(WT_y)
           count+=1
    return wt_x, wt_y



#%% Defining zones
exploded_dev_area = dev_area.explode()
Union_cascade = exploded_dev_area.cascaded_union

#%% group all geometries in a boundary component 
import topfarm
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm import TopFarmProblem
from topfarm.plotting import TurbineTypePlotComponent
from topfarm import SpacingConstraint, XYBoundaryConstraint
from topfarm.constraint_components.boundary import TurbineSpecificBoundaryComp
from topfarm.easy_drivers import EasyRandomSearchDriver, EasyScipyOptimizeDriver
from topfarm.drivers.random_search_driver import randomize_turbine_type, RandomizeTurbineTypeAndPosition
from topfarm.constraint_components.boundary import InclusionZone, ExclusionZone
from py_wake.flow_map import Points
from topfarm import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent 
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian  
from topfarm.plotting import NoPlot, XYPlotComp


#%%

np.transpose((exploded_dev_area.iloc[0,:].geometry.exterior.coords.xy[0], exploded_dev_area.iloc[0,:].geometry.exterior.coords.xy[1]))
zones_xx = []
zones_yy = []

zones=[]
for indx in range(0,len(exploded_dev_area)): 
    zones.append(np.transpose((exploded_dev_area.iloc[indx,:].geometry.exterior.coords.xy[0],
                               exploded_dev_area.iloc[indx,:].geometry.exterior.coords.xy[1])))
                        
    
#%%
zones_inc = []
for indx in range(0,len(zones)):
    zones_inc.append(InclusionZone(zones[indx]))
#zones_inc = [InclusionZone(zones[0]), InclusionZone(zones[1])]    
xybound = XYBoundaryConstraint(zones_inc, boundary_type='multi_polygon')

# RERUN WHEN I NEED TO DO A NEW OPTIMIZATION


#%%
def aep_func(x, y, type, **kwargs):
    simres = wfm(x, y, type=type, **kwargs)
    return simres.aep(normalize_probabilities=True).values.sum()

def daep_func(x, y, type, **kwargs):
    grad = wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'])(x, y)
    return grad


#%%
wfm = IEA37SimpleBastankhahGaussian(site,my_wt)


#%%
min_x=100000000
min_y=100000000
max_x=0
max_y=0
 
for indx in range(0,len(dev_area)):
    if dev_area.iloc[indx,:].geometry.bounds[0]<min_x:
        min_x = dev_area.iloc[indx,:].geometry.bounds[0]
    if dev_area.iloc[indx,:].geometry.bounds[1]<min_y:
        min_y = dev_area.iloc[indx,:].geometry.bounds[1] 
    if dev_area.iloc[indx,:].geometry.bounds[2]>max_x:
        max_x = dev_area.iloc[indx,:].geometry.bounds[2]
    if dev_area.iloc[indx,:].geometry.bounds[3]>max_y:
        max_y = dev_area.iloc[indx,:].geometry.bounds[3]  

x_min, x_max = min_x, max_x # limits for x
y_min, y_max = min_y, max_y # limits for y

n_wd = 12
n_wt = 7
wt_x, wt_y = initial_positions(Union_cascade,n_wt)

aep_comp = CostModelComponent(input_keys=[('x', wt_x), ('y', wt_y)],
                                          n_wt=n_wt,
                                          cost_function=aep_func,
                                          objective=True,
                                          maximize=True,
                                          output_keys=[('aep', 0)],
                                          output_unit='GWh')

def get_aep4smart_start():
    def aep4smart_start(X, Y, wt_x, wt_y):
        sim_res = wfm(wt_x, wt_y, wd=np.arange(0, 360+360/n_wd, 360/n_wd), ws=[6, 8, 10])
        H = np.full(X.shape, 100)
        return sim_res.aep_map(Points(X, Y, H)).values
    return aep4smart_start

aep_comp.smart_start = get_aep4smart_start
    



#%% 


problem = TopFarmProblem(design_vars={'x': wt_x,
                                      'y': wt_y},
                        constraints=[xybound, SpacingConstraint(760)],
                        cost_comp=aep_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=10, tol=0.1),
                        plot_comp=XYPlotComp(),
                        expected_cost=1e-2)


xi = np.linspace(Union_cascade.bounds[0], Union_cascade.bounds[2], 50)
yi = np.linspace(Union_cascade.bounds[1], Union_cascade.bounds[3], 50)
X, Y = np.meshgrid(xi, yi)
x_smart, y_smart = problem.smart_start(X, Y, aep_comp.smart_start(), random_pct=0, seed=1, plot=True)


#%%
_, state, _ = problem.optimize()


#%% 




