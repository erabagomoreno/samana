from samana.forward_model import forward_model
from samana.Data.wgdj0405_JWST import WGDJ0405JWST
from samana.Model.wgdj0405_model import WGDJ0405ModelEPLM3M4Shear
import os
import numpy as np
import sys

# set the job index for the run
job_index = int(sys.argv[1])
data_class = WGDJ0405JWST()
model = WGDJ0405ModelEPLM3M4Shear
preset_model_name = 'WDM'
kwargs_sample_realization = {'log10_sigma_sub': ['UNIFORM', -2.5, -1.0],
                            'log_mc': ['UNIFORM', 4.0, 10.0]
                            }
kwargs_sample_source = {'source_size_pc': ['UNIFORM', 1, 10]}
kwargs_sample_macro_fixed = {
    # 'a4_a': ['FIXED', data_class.a4a_true],
    # 'a3_a': ['FIXED', data_class.a3a_true],
     #'delta_phi_m3': ['FIXED', data_class.delta_phi_m3_true],
    'gamma': ['FIXED', 2.0],
    'a4_a': ['GAUSSIAN', 0.0, 0.01],
    'a3_a': ['GAUSSIAN', 0.0, 0.005],
    'delta_phi_m3': ['UNIFORM', -np.pi/6, np.pi/6]
}

job_name = 'wgdj0405'
use_imaging_data = False
output_path = os.getcwd() + '/data/samana_jobs/'+job_name+'/'
n_keep = 2000
tolerance = np.inf
verbose = True
random_seed_init = None
n_pso_particles = None
n_pso_iterations = None
test_mode = False
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
