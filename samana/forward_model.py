from pyHalo.preset_models import preset_model_from_name
from samana.forward_model_util import filenames, sample_prior, align_realization, \
    flux_ratio_summary_statistic, flux_ratio_likelihood
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution, auto_raytracing_grid_size
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Util.class_creator import create_im_sim
from lenstronomy.LensModel.QuadOptimizer.optimizer import Optimizer
from samana.image_magnification_util import setup_gaussian_source
from samana.param_managers import auto_param_class
from copy import deepcopy
import os
import subprocess
import numpy as np
from time import time


def forward_model(output_path, job_index, n_keep, data_class, model, preset_model_name,
                  kwargs_sample_realization, kwargs_sample_source, kwargs_sample_fixed_macromodel,
                  tolerance, log_mlow_mass_sheets=6.0, n_max_shapelets=None,
                  rescale_grid_size=1.0, rescale_grid_resolution=2.0, readout_macromodel_samples=True,
                  verbose=False, random_seed_init=None, readout_steps=10, write_sampling_rate=True,
                  n_pso_particles=10, n_pso_iterations=50, num_threads=1, astrometric_uncertainty=True,
                  resample_kwargs_lens=False, kde_sampler=None, image_data_grid_resolution_rescale=1.0,
                  use_imaging_data=True, fitting_sequence_kwargs=None, test_mode=False):
    """

    :param output_path:
    :param job_index:
    :param n_keep:
    :param data_class:
    :param model:
    :param preset_model_name:
    :param kwargs_sample_realization:
    :param kwargs_sample_source:
    :param kwargs_sample_fixed_macromodel:
    :param tolerance:
    :param log_mlow_mass_sheets:
    :param n_max_shapelets:
    :param rescale_grid_size:
    :param rescale_grid_resolution:
    :param readout_macromodel_samples:
    :param verbose:
    :param random_seed_init:
    :param readout_steps:
    :param write_sampling_rate:
    :param n_pso_particles:
    :param n_pso_iterations:
    :param num_threads:
    :param astrometric_uncertainty:
    :param resample_kwargs_lens:
    :param kde_sampler:
    :param image_data_grid_resolution_rescale:
    :param use_imaging_data:
    :param fitting_sequence_kwargs:
    :param test_mode:
    :return:
    """

    filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio, \
    filename_macromodel_samples = filenames(output_path, job_index)
    filename_mags2 = output_path + 'job_' + str(job_index) + '/fluxes2.txt'

    # if the required directories do not exist, create them
    if os.path.exists(output_path) is False:
        proc = subprocess.Popen(['mkdir', output_path])
        proc.wait()
    if os.path.exists(output_path + 'job_' + str(job_index)) is False:
        proc = subprocess.Popen(['mkdir', output_path + 'job_' + str(job_index)])
        proc.wait()

    if verbose:
        print('reading output to files: ')
        print(filename_parameters)
        print(filename_mags)
    # You can restart inferences from previous runs by simply running the function again. In the following lines, the
    # code looks for existing output files, and determines how many samples to add based on how much output already
    # exists.
    if os.path.exists(filename_mags):
        _m = np.loadtxt(filename_mags)
        try:
            n_kept = _m.shape[0]
        except:
            n_kept = 1
        write_param_names = False
        write_param_names_macromodel_samples = False
    else:
        n_kept = 0
        _m = None
        write_param_names = True
        write_param_names_macromodel_samples = True

    if n_kept >= n_keep:
        print('\nSIMULATION ALREADY FINISHED.')
        return

    if 'source_size_pc_2' in kwargs_sample_source:
        dual_flag = True
    else:
        dual_flag = False
    # Initialize stuff for the inference
    parameter_array = None
    mags_out = None
    macromodel_samples_array = None
    readout = False
    break_loop = False
    accepted_realizations_counter = 0
    acceptance_rate_counter = 0
    iteration_counter = 0
    acceptance_ratio = np.nan
    sampling_rate = np.nan
    t0 = time()

    if random_seed_init is None:
        # pick a random integer from which to generate random seeds
        random_seed_init = np.random.randint(0, 4294967295)

    if verbose:
        print('starting with ' + str(n_kept) + ' samples accepted, ' + str(n_keep - n_kept) + ' remain')
        print('existing magnifications: ', _m)
        print('samples remaining: ', n_keep - n_kept)
        print('running simulation with a summary statistic tolerance of: ', tolerance)
    # start the simulation, the while loop will execute until one has obtained n_keep samples from the posterior
    seed_counter = 0 + n_kept
    while True:

        # the random seed in numpy maxes out at 4294967295
        random_seed = random_seed_init + seed_counter
        if random_seed > 4294967295:
            random_seed = random_seed - 4294967296

        if dual_flag:
            # RK: added mag2 unpacking
            magnifications, magnifications2, images, realization_samples, source_samples, macromodel_samples, macromodel_samples_fixed, \
            logL_imaging_data, fitting_sequence, stat, stat2, flux_ratio_likelihood_weight, flux_ratio_likelihood_weigh2, bic, \
            param_names_realization, param_names_source, param_names_macro, \
            param_names_macro_fixed, _, _, _ = forward_model_single_iteration(data_class, model, preset_model_name, kwargs_sample_realization,
                                                kwargs_sample_source, kwargs_sample_fixed_macromodel, log_mlow_mass_sheets,
                                                rescale_grid_size, rescale_grid_resolution, image_data_grid_resolution_rescale,
                                                verbose, random_seed, n_pso_particles, n_pso_iterations, num_threads,
                                                n_max_shapelets, astrometric_uncertainty, resample_kwargs_lens, kde_sampler,
                                                                        use_imaging_data, fitting_sequence_kwargs, test_mode)
        else:
            magnifications, images, realization_samples, source_samples, macromodel_samples, macromodel_samples_fixed, \
            logL_imaging_data, fitting_sequence, stat, flux_ratio_likelihood_weight, bic, param_names_realization, param_names_source, param_names_macro, \
            param_names_macro_fixed, _, _, _ = forward_model_single_iteration(data_class, model, preset_model_name, kwargs_sample_realization,
                                                kwargs_sample_source, kwargs_sample_fixed_macromodel, log_mlow_mass_sheets,
                                                rescale_grid_size, rescale_grid_resolution, image_data_grid_resolution_rescale,
                                                verbose, random_seed, n_pso_particles, n_pso_iterations, num_threads,
                                                n_max_shapelets, astrometric_uncertainty, resample_kwargs_lens, kde_sampler,
                                                                        use_imaging_data, fitting_sequence_kwargs, test_mode)

        seed_counter += 1
        acceptance_rate_counter += 1
        # Once we have computed a couple realizations, keep a log of the time it takes to run per realization
        if acceptance_rate_counter == 10:
            time_elapsed = time() - t0
            time_elapsed_minutes = time_elapsed / 60
            sampling_rate = time_elapsed_minutes / acceptance_rate_counter
            readout_sampling_rate = True
        else:
            readout_sampling_rate = False

        # this keeps track of how many realizations were analyzed, and resets after each readout (set by readout_steps)
        # The purpose of this counter is to keep track of the acceptance rate
        iteration_counter += 1
        if stat < tolerance:
            # If the statistic is less than the tolerance threshold, we keep the parameters
            accepted_realizations_counter += 1
            n_kept += 1
            params = np.append(realization_samples, source_samples)
            params = np.append(params, bic)
            params = np.append(params, stat)
            params = np.append(params, flux_ratio_likelihood_weight)
            params = np.append(params, logL_imaging_data)
            params = np.append(params, random_seed)
            param_names = param_names_realization + param_names_source + ['bic', 'summary_statistic', 'flux_ratio_likelihood',
                                                                          'logL_image_data', 'seed']
            acceptance_ratio = accepted_realizations_counter / iteration_counter

            if parameter_array is None:
                parameter_array = params
            else:
                parameter_array = np.vstack((parameter_array, params))
            if dual_flag:
                if mags_out is None:
                    mags_out2 = magnifications2
                else:
                    mags_out2 = np.vstack((mags_out2, magnifications2))
            if mags_out is None:
                mags_out = magnifications
            else:
                mags_out = np.vstack((mags_out, magnifications))
            if macromodel_samples_array is None:
                macromodel_samples_array = np.array(macromodel_samples)
            else:
                macromodel_samples_array = np.vstack((macromodel_samples_array, macromodel_samples))
            if verbose:
                print('N_kept: ', n_kept)
                print('N remaining: ', n_keep - n_kept)

        if verbose:
            print('accepted realizations counter: ', acceptance_rate_counter)
        # readout if either of these conditions are met
        if accepted_realizations_counter == readout_steps:
            readout = True
            if verbose:
                print('reading out data on this iteration.')
            accepted_realizations_counter = 0
            iteration_counter = 0
        # break loop if we have collected n_keep samples
        if n_kept == n_keep:
            readout = True
            break_loop = True
            if verbose:
                print('final data readout...')
        if readout_sampling_rate and write_sampling_rate:
            with open(filename_sampling_rate, 'w') as f:
                f.write(str(np.round(sampling_rate, 2)) + ' ')
                f.write('\n')

        if readout:
            # Now write stuff to file
            readout = False
            with open(filename_acceptance_ratio, 'a') as f:
                f.write(str(np.round(acceptance_ratio, 8)) + ' ')
                f.write('\n')
            if verbose:
                print('writing parameter output to ' + filename_parameters)
            with open(filename_parameters, 'a') as f:
                if write_param_names:
                    param_name_string = ''
                    for name in param_names:
                        param_name_string += name + ' '
                    f.write(param_name_string + '\n')
                    write_param_names = False

                nrows, ncols = int(parameter_array.shape[0]), int(parameter_array.shape[1])
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(parameter_array[row, col], 7)) + ' ')
                    f.write('\n')
            if verbose:
                print('writing flux ratio output to ' + filename_mags)
            with open(filename_mags, 'a') as f:
                nrows, ncols = int(mags_out.shape[0]), int(mags_out.shape[1])
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(mags_out[row, col], 5)) + ' ')
                    f.write('\n')
            if dual_flag:
                if verbose:
                    print('writing flux ratio output to ' +filename_mags2)
                with open(filename_mags2, 'a') as f:
                    nrows, ncols = int(mags_out2.shape[0]), int(mags_out2.shape[1])
                    for row in range(0, nrows):
                        for col in range(0, ncols):
                            f.write(str(np.round(mags_out2[row, col], 5)) + ' ')
                        f.write('\n')

            if readout_macromodel_samples:
                if verbose:
                    print('writing macromodel samples to ' + filename_macromodel_samples)
                nrows, ncols = int(macromodel_samples_array.shape[0]), int(macromodel_samples_array.shape[1])
                with open(filename_macromodel_samples, 'a') as f:
                    if write_param_names_macromodel_samples:
                        param_name_string = ''
                        for name in param_names_macro:
                            param_name_string += name + ' '
                        f.write(param_name_string + '\n')
                        write_param_names_macromodel_samples = False
                    for row in range(0, nrows):
                        for col in range(0, ncols):
                            f.write(str(np.round(macromodel_samples_array[row, col], 5)) + ' ')
                        f.write('\n')
            parameter_array = None
            mags_out = None
            macromodel_samples_array = None

        if break_loop:
            print('\nSIMULATION FINISHED')
            break

def forward_model_single_iteration(data_class, model, preset_model_name, kwargs_sample_realization,
                            kwargs_sample_source, kwargs_sample_macro_fixed, log_mlow_mass_sheets=6.0, rescale_grid_size=1.0,
                            rescale_grid_resolution=2.0, image_data_grid_resolution_rescale=1.0, verbose=False, seed=None,
                                   n_pso_particles=10, n_pso_iterations=50, num_threads=1, n_max_shapelets=None, astrometric_uncertainty=True,
                                   resample_kwargs_lens=False, kde_sampler=None, use_imaging_data=True,
                                   fitting_kwargs_list=None,
                                   test_mode=False):

    # set the random seed for reproducibility
    np.random.seed(seed)
    if astrometric_uncertainty:
        delta_x_image, delta_y_image = data_class.perturb_image_positions()
    else:
        delta_x_image, delta_y_image = np.zeros(len(data_class.x_image)), np.zeros(len(data_class.y_image))
    model_class = model(data_class, kde_sampler, shapelets_order=n_max_shapelets)
    realization_dict, realization_samples, realization_param_names = sample_prior(kwargs_sample_realization)
    source_dict, source_samples, source_param_names = sample_prior(kwargs_sample_source)
    if 'source_size_pc_2' in source_dict:
        dual_flag = True
    else:
        dual_flag = False
    macromodel_samples_fixed_dict, samples_macromodel_fixed, param_names_macro_fixed = sample_prior(kwargs_sample_macro_fixed)
    if 'SUBSTRUCTURE_REALIZATION' in realization_param_names:
        if verbose: print('using a user-specified dark matter realization')
        realization_init = realization_dict['SUBSTRUCTURE_REALIZATION']
        preset_realization = True
    else:
        present_model_function = preset_model_from_name(preset_model_name)
        realization_init = present_model_function(data_class.z_lens, data_class.z_source, **realization_dict)
        preset_realization = False
    if verbose:
        print('random seed: ', seed)
        print('SOURCE PARAMETERS: ')
        print(source_dict)
        print('REALIZATION PARAMETERS: ')
        print(realization_dict)
        print('FIXED MACROMODEL SAMPLES: ')
        print(macromodel_samples_fixed_dict)
    if resample_kwargs_lens:
        if model_class.kde_sampler is None:
            raise Exception('if resample_kwargs_lens is True, the model class must be instantiated with an instance of'
                            'KwargsLensSampler.')
        else:
            kwargs_lens_macro_init, _, _ = model_class.kde_sampler.draw()
            if verbose:
                print('starting point for PSO: ', kwargs_lens_macro_init)
    else:
        kwargs_lens_macro_init = None
    kwargs_params = model_class.kwargs_params(kwargs_lens_macro_init=kwargs_lens_macro_init,
                                              delta_x_image=-delta_x_image,
                                              delta_y_image=-delta_y_image,
                                              macromodel_samples_fixed=macromodel_samples_fixed_dict)
    pixel_size = data_class.coordinate_properties[0] / data_class.kwargs_numerics['supersampling_factor']
    kwargs_model_align, _, _, _ = model_class.setup_kwargs_model(
        decoupled_multiplane=False,
        macromodel_samples_fixed=macromodel_samples_fixed_dict)
    kwargs_lens_align = kwargs_params['lens_model'][0]
    if preset_realization:
        realization = realization_init
    else:
        realization, _, _, lens_model_align, _ = align_realization(realization_init, kwargs_model_align['lens_model_list'],
                                    kwargs_model_align['lens_redshift_list'], kwargs_lens_align,
                                    data_class.x_image,
                                    data_class.y_image)
    lens_model_list_halos, redshift_list_halos, kwargs_halos, _ = realization.lensing_quantities(
        kwargs_mass_sheet={'log_mlow_sheets': log_mlow_mass_sheets})
    if verbose:
        print('realization has '+str(len(realization.halos))+' halos')
    grid_resolution_image_data = pixel_size * image_data_grid_resolution_rescale
    astropy_cosmo = realization.lens_cosmo.cosmo.astropy

    kwargs_model, lens_model_init, kwargs_lens_init, index_lens_split = model_class.setup_kwargs_model(
        decoupled_multiplane=True,
        lens_model_list_halos=lens_model_list_halos,
        grid_resolution=grid_resolution_image_data,
        redshift_list_halos=list(redshift_list_halos),
        kwargs_halos=kwargs_halos,
        verbose=verbose,
        macromodel_samples_fixed=macromodel_samples_fixed_dict)
    kwargs_constraints = model_class.kwargs_constraints
    kwargs_likelihood = model_class.kwargs_likelihood
    if astrometric_uncertainty:
        kwargs_constraints['point_source_offset'] = True
    else:
        kwargs_constraints['point_source_offset'] = False

    if use_imaging_data:
        if verbose:
            print('running fitting sequence...')
            t0 = time()
        fitting_sequence = FittingSequence(data_class.kwargs_data_joint,
                                           kwargs_model,
                                           kwargs_constraints,
                                           kwargs_likelihood,
                                           kwargs_params,
                                           mpi=False, verbose=verbose)
        if fitting_kwargs_list is None:
            fitting_kwargs_list = [
                ['PSO', {'sigma_scale': 1., 'n_particles': n_pso_particles, 'n_iterations': n_pso_iterations,
                         'threadCount': num_threads}]
            ]
        chain_list = fitting_sequence.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_sequence.best_fit()
        if verbose:
            print('done in ' + str(time() - t0) + ' seconds')
            likelihood_module = fitting_sequence.likelihoodModule
            print(likelihood_module.log_likelihood(kwargs_result, verbose=True))
        kwargs_solution = kwargs_result['kwargs_lens']
        lens_model = LensModel(lens_model_list=kwargs_model['lens_model_list'],
                               lens_redshift_list=kwargs_model['lens_redshift_list'],
                               multi_plane=kwargs_model['multi_plane'],
                               decouple_multi_plane=kwargs_model['decouple_multi_plane'],
                               kwargs_multiplane_model=kwargs_model['kwargs_multiplane_model'],
                               z_source=kwargs_model['z_source'])

    else:
        param_class = auto_param_class(lens_model_init.lens_model_list,
                                       kwargs_lens_align,
                                       macromodel_samples_fixed_dict)
        kwargs_lens_init = kwargs_lens_align + kwargs_lens_init[len(kwargs_lens_align):]
        opt = Optimizer.decoupled_multiplane(data_class.x_image,
                                             data_class.y_image,
                                             lens_model_init,
                                             kwargs_lens_init,
                                             index_lens_split,
                                             param_class,
                                             tol_simplex_func=1e-5,
                                             simplex_n_iterations=500
                                             )
        kwargs_solution, _ = opt.optimize(20, 50, verbose=verbose, seed=seed)
        lens_model = LensModel(lens_model_list=kwargs_model['lens_model_list'],
                               lens_redshift_list=kwargs_model['lens_redshift_list'],
                               multi_plane=kwargs_model['multi_plane'],
                               decouple_multi_plane=kwargs_model['decouple_multi_plane'],
                               kwargs_multiplane_model=opt.kwargs_multiplane_model,
                               z_source=kwargs_model['z_source'])

    if verbose:
        print('\n')
        print('kwargs solution: ', kwargs_solution)
        print('\n')
        print('computing image magnifications...')
    t0 = time()

    source_x, source_y = lens_model.ray_shooting(data_class.x_image, data_class.y_image,
                                                 kwargs_solution)
    source_model_quasar, kwargs_source = setup_gaussian_source(source_dict['source_size_pc'],
                                                               np.mean(source_x), np.mean(source_y),
                                                               astropy_cosmo, data_class.z_source)


    grid_size = rescale_grid_size * auto_raytracing_grid_size(source_dict['source_size_pc'])

    grid_resolution = rescale_grid_resolution * auto_raytracing_grid_resolution(source_dict['source_size_pc'])

    magnifications, images = model_class.image_magnification_gaussian(source_model_quasar,
                                                                      kwargs_source,
                                                                      lens_model_init,
                                                                      kwargs_lens_init,
                                                                      kwargs_solution,
                                                                      grid_size, grid_resolution)

    if dual_flag:
        #copying these lines from above but four source size 2
        source_model_quasar2, kwargs_source2 = setup_gaussian_source(source_dict['source_size_pc_2'], #RK in the script make sure it uses source_size_pc and _2 keywords
                                                                   np.mean(source_x), np.mean(source_y),
                                                                   astropy_cosmo, data_class.z_source)

        grid_size2 = rescale_grid_size * auto_raytracing_grid_size(source_dict['source_size_pc_2']) #RK adding this

        grid_resolution2 = rescale_grid_resolution * auto_raytracing_grid_resolution(source_dict['source_size_pc_2'])

        magnifications2, images = model_class.image_magnification_gaussian(source_model_quasar2,
                                                                          kwargs_source2,
                                                                          lens_model_init,
                                                                          kwargs_lens_init,
                                                                          kwargs_solution,
                                                                          grid_size2, grid_resolution2)
    tend = time()
    if verbose:
        print('computed magnifications in '+str(np.round(tend - t0, 1))+' seconds')
        print('magnifications: ', magnifications,'magnifications2: ', magnifications2) #ER: adding a second print for mags2

    samples_macromodel = []
    param_names_macro = []
    for lm in kwargs_solution:
        for key in lm.keys():
            samples_macromodel.append(lm[key])
            param_names_macro.append(key)
    samples_macromodel = np.array(samples_macromodel)

    if use_imaging_data:
        bic = fitting_sequence.bic
        image_model = create_im_sim(data_class.kwargs_data_joint['multi_band_list'],
                                    data_class.kwargs_data_joint['multi_band_type'],
                                    kwargs_model,
                                    bands_compute=None,
                                    image_likelihood_mask_list=[data_class.likelihood_mask_imaging_weights],
                                    band_index=0,
                                    kwargs_pixelbased=None,
                                    linear_solver=True)
        logL_imaging_data = image_model.likelihood_data_given_model(kwargs_result['kwargs_lens'],
                kwargs_result['kwargs_source'],
                kwargs_result['kwargs_lens_light'],
                kwargs_result['kwargs_ps'],
                kwargs_extinction=kwargs_result['kwargs_extinction'],
                kwargs_special=kwargs_result['kwargs_special'],
                source_marg=False,
                linear_prior=None,
                check_positive_flux=False)[0]
        if verbose:
            logL_imaging_data_no_custom_mask = fitting_sequence.likelihoodModule.image_likelihood.logL(**kwargs_result)[
                0]
            print('imaging data likelihood (without custom mask): ', logL_imaging_data_no_custom_mask)
            print('imaging data likelihood (with custom mask): ', logL_imaging_data)
    else:
        bic = -1000
        logL_imaging_data = -1000
        # here we replace the lens model used to solve for the four quasar point sources with a lens model that
        # is defined across the entrie image plane. This is useful for visualizing the kappa maps, but is not strictly
        # necessary to run the code.
        lens_model = LensModel(lens_model_list=kwargs_model['lens_model_list'],
                               lens_redshift_list=kwargs_model['lens_redshift_list'],
                               multi_plane=kwargs_model['multi_plane'],
                               decouple_multi_plane=kwargs_model['decouple_multi_plane'],
                               kwargs_multiplane_model=kwargs_model['kwargs_multiplane_model'],
                               z_source=kwargs_model['z_source'])

    stat, flux_ratios, flux_ratios_data = flux_ratio_summary_statistic(data_class.magnifications,
                                                                       magnifications,
                                                                       data_class.flux_uncertainty,
                                                                       data_class.keep_flux_ratio_index,
                                                                       data_class.uncertainty_in_fluxes)

    flux_ratio_likelihood_weight = flux_ratio_likelihood(data_class.magnifications, magnifications,
                                                         data_class.flux_uncertainty, data_class.uncertainty_in_fluxes,
                                                         data_class.keep_flux_ratio_index)

    if dual_flag:
        #copying these lines from above but for magnifications2
        #RK: no flux_ratios_data2 implemented yet
        # it looks like the flux_ratios are never saved just printed below
        stat2, flux_ratios2, flux_ratios_data = flux_ratio_summary_statistic(data_class.magnifications,
                                                                           magnifications2,
                                                                           data_class.flux_uncertainty,
                                                                           data_class.keep_flux_ratio_index,
                                                                           data_class.uncertainty_in_fluxes)

        # no data_class2 yet
        flux_ratio_likelihood_weight2 = flux_ratio_likelihood(data_class.magnifications, magnifications2,
                                                             data_class.flux_uncertainty, data_class.uncertainty_in_fluxes,
                                                             data_class.keep_flux_ratio_index)

    if verbose:
        print('flux ratios data: ', flux_ratios_data)
        print('flux ratios model: ', flux_ratios)
        print('statistic: ', stat)
        print('flux_ratio_likelihood_weight', flux_ratio_likelihood_weight)
        if dual_flag:
            print('flux ratios model 2: ', flux_ratios2)
            print('statistic2: ', stat2)
            print('flux_ratio_likelihood_weight2', flux_ratio_likelihood_weight2)
        if use_imaging_data:
            print('BIC: ', bic)
    if use_imaging_data:
        kwargs_model_plot = {'multi_band_list': data_class.kwargs_data_joint['multi_band_list'],
                         'kwargs_model': kwargs_model,
                         'kwargs_params': kwargs_result}
    else:
        kwargs_model_plot = {}
        fitting_sequence = None

    if test_mode:

        if use_imaging_data is False:
            fitting_sequence = FittingSequence(data_class.kwargs_data_joint,
                                               kwargs_model,
                                               kwargs_constraints,
                                               kwargs_likelihood,
                                               kwargs_params,
                                               mpi=False, verbose=verbose)
            kwargs_result = fitting_sequence.best_fit()
            kwargs_result['kwargs_lens'] = kwargs_solution

        from lenstronomy.Plots.model_plot import ModelPlot
        from lenstronomy.Plots import chain_plot
        import matplotlib.pyplot as plt
        fig = plt.figure(1)
        fig.set_size_inches(16,8)
        ax1 = plt.subplot(141)
        ax2 = plt.subplot(142)
        ax3 = plt.subplot(143)
        ax4 = plt.subplot(144)
        axes_list = [ax1, ax2, ax3, ax4]
        for mag, ax, image in zip(magnifications, axes_list, images):
            ax.imshow(image, origin='lower')
            ax.annotate('magnification: '+str(np.round(mag,2)), xy=(0.3,0.9),
                        xycoords='axes fraction',color='w',fontsize=12)
        plt.show()
        if dual_flag:#ER: Adding mag plot for mag2
            fig = plt.figure(1)
            fig.set_size_inches(16, 8)
            ax1 = plt.subplot(141)
            ax2 = plt.subplot(142)
            ax3 = plt.subplot(143)
            ax4 = plt.subplot(144)
            axes_list = [ax1, ax2, ax3, ax4]
            for mag, ax, image in zip(magnifications2, axes_list, images):
                ax.imshow(image, origin='lower')
                ax.annotate('magnification: ' + str(np.round(mag, 2)), xy=(0.3, 0.9),
                            xycoords='axes fraction', color='w', fontsize=12)
            plt.show()

        modelPlot = ModelPlot(data_class.kwargs_data_joint['multi_band_list'],
                              kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                              fast_caustic=True,
                              image_likelihood_mask_list=[data_class.likelihood_mask_imaging_weights])
        if use_imaging_data:
            chain_plot.plot_chain_list(chain_list, 0)

        fig = plt.figure()
        fig.set_size_inches(6, 6)
        ax = plt.subplot(111)
        kwargs_plot = {'ax': ax,
                       'index_macromodel': list(np.arange(0, len(kwargs_result['kwargs_lens']))),
                       'with_critical_curves': True,
                       'v_min': -0.1, 'v_max': 0.1,
                       'super_sample_factor': 5}
        modelPlot.substructure_plot(band_index=0, **kwargs_plot)
        plt.show()
        a=input('')
    if dual_flag:
        return magnifications, magnifications2, images, realization_samples, source_samples, samples_macromodel, samples_macromodel_fixed, \
               logL_imaging_data, fitting_sequence, \
               stat, stat2, flux_ratio_likelihood_weight, flux_ratio_likelihood_weight2, bic, realization_param_names, \
               source_param_names, param_names_macro, \
               param_names_macro_fixed, kwargs_model_plot, lens_model, kwargs_solution
    else:
        return magnifications, images, realization_samples, source_samples, samples_macromodel, samples_macromodel_fixed, \
               logL_imaging_data, fitting_sequence, \
               stat, flux_ratio_likelihood_weight, bic, realization_param_names, \
               source_param_names, param_names_macro, \
               param_names_macro_fixed, kwargs_model_plot, lens_model, kwargs_solution
