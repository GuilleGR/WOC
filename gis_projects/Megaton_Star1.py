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
#%%  SAVING 
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

#Projecting to UTM. Zone 14N
gdf = gdf.to_crs({'init':'epsg:32614'})

#%% 
exploded_dev_area = gdf.explode()
Union_cascade = exploded_dev_area.cascaded_union
#%% SITE 

freq = obj[2]
new_freq = [x/sum(freq) for x in freq]

site = UniformWeibullSite(p_wd = new_freq,                         # sector frequencies
                          a = obj[0],  # Weibull scale parameter
                          k = obj[1],     # Weibull shape parameter
                          ti = 0.1                                          # turbulence intensity, optional
                         )


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
                    powerCtFunction=PowerCtTabular(wtg_B.WS ,wtg_B.Power,'kW',wtg_B.CT))

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


#%% WIND TURBINES

from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt


windTurbines = WindTurbines(names=['V136 4.5', 'V172 7.2'],diameters=[136, 172],
                            hub_heights=[112, 114],
                            powerCtFunctions = [PowerCtTabular(wtg_A.WS ,wtg_A.Power,'kW',wtg_A.CT),
                                                PowerCtTabular(wtg_B.WS ,wtg_B.Power,'kW',wtg_B.CT)])



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


#%%  Initial positions

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




#%%


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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


min_x=100000000
min_y=100000000
max_x=0
max_y=0

for indx in range(0,len(exploded_dev_area)):
    if exploded_dev_area.iloc[indx,:].geometry.bounds[0]<min_x:
        min_x = exploded_dev_area.iloc[indx,:].geometry.bounds[0]
    if exploded_dev_area.iloc[indx,:].geometry.bounds[1]<min_y:
        min_y = exploded_dev_area.iloc[indx,:].geometry.bounds[1] 
    if exploded_dev_area.iloc[indx,:].geometry.bounds[2]>max_x:
        max_x = exploded_dev_area.iloc[indx,:].geometry.bounds[2]
    if exploded_dev_area.iloc[indx,:].geometry.bounds[3]>max_y:
        max_y = exploded_dev_area.iloc[indx,:].geometry.bounds[3]  

x_min, x_max = min_x, max_x # limits for x
y_min, y_max = min_y, max_y # limits for y

wt_x,wt_y = initial_positions(Union_cascade, 45)


#%% WIND FARM MODEL 

windFarmModel = IEA37SimpleBastankhahGaussian(site,windTurbines)

#%%
n_wt = 45
init_types = 10 * [1] + 15 * [0] + 10 *[1] + 10 *[0] 

tf2 = TopFarmProblem(
    design_vars={'x': wt_x,'y': wt_y},
    cost_comp=PyWakeAEPCostModelComponent(windFarmModel, n_wt, additional_input=[('type', np.zeros(n_wt))], grad_method=None),
    driver=EasyScipyOptimizeDriver(maxiter=50),
    constraints=[xybound, SpacingConstraint(760)],
    plot_comp=XYPlotComp())
tf2['type']=init_types

x = np.linspace(x_min,x_max,500)
y = np.linspace(y_min,y_max,500)
YY, XX = np.meshgrid(y, x)


#%%
tf2.smart_start(XX, YY, tf.cost_comp.get_aep4smart_start(type=init_types))
cost3, state3 = tf2.evaluate()

#%% Optimizate
cost4, state4, recorder4 = tf2.optimize()



#%%

#aep_comp = CostModelComponent(input_keys=[('x', wt_x), ('y', wt_y)],
 #                                         n_wt=n_wt,
  #                                        cost_function=aep_func,
   #                                       objective=True,
    #                                      maximize=True,
     #                                     output_keys=[('aep', 0)],
      #                                    output_unit='GWh')

n_wt = 30
#site2 = IEA37Site(n_wt)
wfm = IEA37SimpleBastankhahGaussian(site,windTurbines)

#%%

n_wd = 12
n_wt = 60
wt_x, wt_y = initial_positions(Union_cascade,n_wt)

#%%

def aep_func(x, y, type, **kwargs):
    simres = wfm(x, y, type=type, **kwargs)
    return simres.aep(normalize_probabilities=True).values.sum()

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

init_types = 5 * [1] + 6 * [0] 

problem = TopFarmProblem(design_vars={'x': wt_x,
                                      'y': wt_y},
                        constraints=[xybound, SpacingConstraint(760)],
                        cost_comp=PyWakeAEPCostModelComponent(wfm, n_wt, additional_input=[('type', np.zeros(n_wt))], grad_method=None),
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=10, tol=0.1),
                        plot_comp=XYPlotComp(),
                        expected_cost=1e-2)
tf['type']=init_types

#%%

xi = np.linspace(Union_cascade.bounds[0], Union_cascade.bounds[2], 50)
yi = np.linspace(Union_cascade.bounds[1], Union_cascade.bounds[3], 50)
X, Y = np.meshgrid(xi, yi)



#%%

xi = np.linspace(Union_cascade.bounds[0], Union_cascade.bounds[2], 50)
yi = np.linspace(Union_cascade.bounds[1], Union_cascade.bounds[3], 50)
X, Y = np.meshgrid(xi, yi)
x_smart, y_smart = problem.smart_start(X, Y, aep_comp.smart_start(), random_pct=0, seed=1, plot=True)



    











