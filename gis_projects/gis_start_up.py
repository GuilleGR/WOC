#!/usr/bin/env python
# coding: utf-8

# ### Create a directory to store GIS project and get libraries

# In[1]:


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import rasterio
from arcgis.features import FeatureLayer
from arcgis.gis import GIS
from arcgis.features import FeatureLayerCollection
from arcgis.geometry import Geometry
from arcgis.features import Feature, FeatureSet


# make directory called 'gis_projects' in c drive if not exists

if not os.path.exists('C:\\Users\\Guillermo Rilova\\OneDrive - Greengo Energy\\Documents\\Wind\DEVELOPMENT\R&D\\gis_projects'):
    os.makedirs('C:\\Users\\Guillermo Rilova\\OneDrive - Greengo Energy\\Documents\\Wind\DEVELOPMENT\R&D\\gis_projects')


# ! use pip install arcgis --user if have problem




# ### function to grab data from DK

# In[2]:


class ArcGISHandler:
    def __init__(self):
        # Read the configuration from a JSON file
        # with open(r'D:/Python/Production/Config_files/config.json', 'r') as f:
        #     config = json.load(f)

        # # Extract GIS configuration and initialize GIS object
        # gis_config = config['online_gis']
        # self.gis = GIS(gis_config['url'], gis_config['username'], gis_config['password'])

        # ! Recommend putting details into JSON as above - switch out with Thomas' details for now (I already share mine with Mara - there is a limit to how many people can share a single account)
        self.gis = GIS('https://greengo-eu.maps.arcgis.com', 'GGE_RT', 'ARcgispro13*')

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
        


# example of how to use the class
# project_url = 'https://services-eu1.arcgis.com/Iu9KJRhfAInbTSB3/arcgis/rest/services/Denmark_Projects_1f1dd/FeatureServer/0'ArithmeticError
# arcgis_handler = ArcGISHandler()
# gdf = arcgis_handler.fetch_data_gdf(project_url)



# ### get some data
# 

# In[3]:


### get DK projects
project_url = 'https://services-eu1.arcgis.com/Iu9KJRhfAInbTSB3/arcgis/rest/services/Denmark_Projects_1f1dd/FeatureServer/0'
arcgis_handler = ArcGISHandler()
project_dk_gdf = arcgis_handler.fetch_data_gdf(project_url)



windfarms_url = 'https://services-eu1.arcgis.com/Iu9KJRhfAInbTSB3/arcgis/rest/services/Windmills_2021/FeatureServer/0'
arcgis_handler = ArcGISHandler()
windfarms_dk_gdf = arcgis_handler.fetch_data_gdf(windfarms_url)


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


# ### Do something

# In[4]:


# lets take first 10 wind turbines
windfarms_dk_example_gdf = windfarms_dk_gdf.head(10)
windfarms_dk_example_gdf.plot()

# keep Tilsluttet	Kapa_kW	Rotor_Dia	NavHoej	TotalHoej	Fabrikat	Model
windfarms_dk_example_gdf = windfarms_dk_example_gdf[['Tilsluttet', 'Kapa_kW', 'Rotor_Dia', 'NavHoej', 'TotalHoej', 'Fabrikat', 'Model', 'geometry']]

# dtypes
windfarms_dk_example_gdf.dtypes

# which dk projects are within TotalHoej x 1.7 of windfarms
windfarms_dk_example_gdf['buffer_distance'] = windfarms_dk_example_gdf['TotalHoej'] * 1.7

windfarms_dk_example_gdf.head(10)

# FOR SOME reason there are wind turbines with Null geometry - remove them
windfarms_dk_example_gdf = windfarms_dk_example_gdf[windfarms_dk_example_gdf['geometry'].notna()]


windfarms_dk_example_gdf_area = windfarms_dk_example_gdf.copy() 

# # Apply the buffer method to each geometry
windfarms_dk_example_gdf_area['geometry'] = windfarms_dk_example_gdf_area.apply(lambda row: row['geometry'].buffer(row['buffer_distance']), axis=1)

#Plot the buffered geometries
windfarms_dk_example_gdf_area['geometry'].plot()





# In[50]:


project_dk_gdf.head(30)


# ### where projects are within buffer of windfarms

# In[56]:


import geopandas as gpd

# get Project_Name = M75 - Hølletvej
project_dk_gdf_example = project_dk_gdf[project_dk_gdf['Project_Name'] == 'MXX - Lundsgaard']

# do a plot with extent increased by 1000m and add in windfarms
#  Calculate the bounds of the project area and expand by 1000m
minx, miny, maxx, maxy = project_dk_gdf_example.total_bounds
buffer = 10000  # 1000 meters buffer
bounds = [minx - buffer, miny - buffer, maxx + buffer, maxy + buffer]

# Create a plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the project area
project_dk_gdf_example.plot(ax=ax, color='blue', edgecolor='k')

# Plot the windfarms - ORIGINAL WINDFARMS
windfarms_dk_gdf.plot(ax=ax, color='green', edgecolor='k')

# Set the plot limits to the calculated bounds
ax.set_xlim([bounds[0], bounds[2]])
ax.set_ylim([bounds[1], bounds[3]])

# Optional: Add grid, labels, title, etc.
ax.grid(True)
ax.set_title('Project Area and Windfarms')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show the plot
plt.show()


#%% get Project_Name = M75 - Hølletvej

project_dk_gdf_example = project_dk_gdf[project_dk_gdf['Project_Name'] == 'M146 - Brønderslev Nord']

# do a plot with extent increased by 1000m and add in windfarms
#  Calculate the bounds of the project area and expand by 1000m
minx, miny, maxx, maxy = project_dk_gdf_example.total_bounds
buffer = 10000  # 1000 meters buffer
bounds = [minx - buffer, miny - buffer, maxx + buffer, maxy + buffer]

# Create a plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the project area
project_dk_gdf_example.plot(ax=ax, color='blue', edgecolor='k')

# Plot the windfarms - ORIGINAL WINDFARMS
windfarms_dk_gdf.plot(ax=ax, color='green', edgecolor='k')

# Set the plot limits to the calculated bounds
ax.set_xlim([bounds[0], bounds[2]])
ax.set_ylim([bounds[1], bounds[3]])

# Optional: Add grid, labels, title, etc.
ax.grid(True)
ax.set_title('Project Area and Windfarms')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show the plot
plt.show()




