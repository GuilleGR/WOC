import numpy as np
import pandas as pd 
import xarray as xr
import time as tm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.ground_models.ground_models import Mirror
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.flow_map import Points
from py_wake.site.shear import PowerShear
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.plotting import XYPlotComp
from topfarm import SpacingConstraint

# =============================================================================
# Site and wind turbine model 
# =============================================================================
n_wd = 180
site = 'define your site'
windTurbines = 'define your turbine curve'

# =============================================================================
# Define TurbOPark (original and corrected, no blockage)
# =============================================================================

wfm = 'define your wake model'
# =============================================================================
# L64 COP - potential positions and number of wind turbines desired
# =============================================================================
wt_x = 'define initial positions - they can be randomized'
wt_y = 'idem'

n_wt = len(wt_x)

# AEP function and gradients (with autograd)
def aep_func(x, y):
    simres = wfm(x, y)
    return simres.aep(normalize_probabilities=True).values.sum()

# AEP cost component
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
        H = np.full(X.shape, wfm.windTurbines.hub_height())
        return sim_res.aep_map(Points(X, Y, H)).values
    return aep4smart_start

aep_comp.smart_start = get_aep4smart_start
    
problem = TopFarmProblem(design_vars={'x': wt_x,
                                      'y': wt_y},
                        constraints=[SpacingConstraint(760)],
                        cost_comp=aep_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=10, tol=0.1),
                        plot_comp=XYPlotComp(),
                        expected_cost=1e-2)

xi = np.linspace(xbounds[0], xbounds[1], 50)
yi = np.linspace(ybounds[0], ybounds[1], 50)
X, Y = np.meshgrid(xi, yi)
x_smart, y_smart = problem.smart_start(X, Y, aep_comp.smart_start(), random_pct=0, seed=1, plot=True)

cost, state, recorder = problem.optimize()   







