from samana.forward_model import forward_model
from samana.Data.Mocks_dual.mock_4_data_dual import Mock4Data_dual
from samana.Model.Mocks_dual.mock_4_model_dual import Mock4Model_dual
import os
import numpy as np
import sys



job_index = int(sys.argv[1])
data_class = Mock4Data_dual()
model = Mock4Model_dual
preset_model_name = 'WDM'
kwargs_sample_realization = {'log10_sigma_sub': ['UNIFORM', -2.5, -1.0],
                            'log_mc': ['UNIFORM', 4.0, 10.0],
                            'LOS_normalization': ['UNIFORM', 0.8,1.2],
                            'shmf_log_slope': ['GAUSSIAN',-1.9,0.05],
                            'truncation_model_subhalos': ['FIXED', 'TRUNCATION_GALACTICUS'], # specifies the tidal truncation model
                            'host_scaling_factor': ['FIXED', 0.5], # formerly k1
                            'redshift_scaling_factor': ['FIXED', 0.3], # formerly k2
                            'cone_opening_angle_arcsec': ['FIXED', 6.0]
                             }

kwargs_sample_source = {'source_size_pc': ['UNIFORM', 1, 10],
                        'source_size_pc_2': ['UNIFORM',40,80]}
kwargs_sample_macro_fixed = {
        'gamma': ['GAUSSIAN', 2.0, 0.1],
    'a4_a': ['GAUSSIAN', 0.0, 0.01],
    'a3_a': ['GAUSSIAN', 0.0, 0.005],
    'delta_phi_m3': ['UNIFORM', -np.pi/6, np.pi/6],
    'delta_phi_m4': ['UNIFORM', -np.pi/8, np.pi/8]
}

job_name = 'mock4_dual'
use_imaging_data = False
output_path = os.getcwd() + '/data/chains/'+job_name+'/'
n_keep = 1000
tolerance = np.inf
verbose = True
random_seed_init = None
n_pso_particles = None
n_pso_iterations = None
test_mode = False
num_threads = 1 #ER: What does this do?
forward_model(output_path, job_index, n_keep, data_class, model, preset_model_name,
                  kwargs_sample_realization, kwargs_sample_source, kwargs_sample_macro_fixed,
               tolerance, random_seed_init=random_seed_init,
              rescale_grid_resolution=2.0,
              # rescale_grid_resolution=2 lowers the resolution of the ray-tracing grid, which makes the calcuation
              # faster without a significant loss of precision as far as I can tell
              verbose=verbose, n_pso_particles=n_pso_particles,
              n_pso_iterations=n_pso_iterations, num_threads=num_threads,
              test_mode=test_mode, use_imaging_data=use_imaging_data)
