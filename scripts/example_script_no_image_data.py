from samana.forward_model import forward_model
from samana.Data.Mocks.mock_1_data import Mock1Data
from samana.Model.Mocks.mock_1_model import Mock1Model
import os
import numpy as np
import sys

# set the job index for the run
job_index = int(sys.argv[1])
data_class = Mock1Data(cosmos_source=True)
model = Mock1Model

job_name = '1606'
preset_model_name = 'WDM' # uses preset models in pyHalo

# Priors on dark matter parameters
kwargs_sample_realization = {'log10_sigma_sub': ['UNIFORM', -2.5, -1.0],
                            'log_mc': ['UNIFORM', 4.0, 10.0]}
# prior on the source size
kwargs_sample_source = {'source_size_pc': ['UNIFORM', 1, 10]}
# prior on the macromodel; here we sample a4_a, a3_a, and the relative orientation of the a3 term
# the orientation of the a4 term is fixed to that of the EPL
kwargs_sample_macro_fixed = {
     'a4_a': ['GAUSSIAN', 0.0, 0.01],
     'a3_a': ['GAUSSIAN', 0.0, 0.005],
     'delta_phi_m3': ['UNIFORM', -np.pi/6, np.pi/6]}
use_imaging_data = False
output_path = os.getcwd() + '/'+job_name+'/'
n_keep = 2000
tolerance = np.inf
verbose = True
random_seed_init = None
n_pso_particles = None
n_pso_iterations = None
test_mode = True
num_threads = 1
forward_model(output_path, job_index, n_keep, data_class, model, preset_model_name,
                  kwargs_sample_realization, kwargs_sample_source, kwargs_sample_macro_fixed,
               tolerance, random_seed_init=random_seed_init,
              rescale_grid_resolution=2.0,
              # rescale_grid_resolution=2 lowers the resolution of the ray-tracing grid, which makes the calcuation
              # faster without a significant loss of precision as far as I can tell
              verbose=verbose, n_pso_particles=n_pso_particles,
              n_pso_iterations=n_pso_iterations, num_threads=num_threads,
              test_mode=test_mode, use_imaging_data=use_imaging_data)
