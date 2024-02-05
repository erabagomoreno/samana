import numpy as np
from copy import deepcopy
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from samana.image_magnification_util import perturbed_fluxes_from_fluxes, perturbed_flux_ratios_from_flux_ratios
from samana.output_storage import Output
import matplotlib.pyplot as plt
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot
from trikde.pdfs import IndepdendentLikelihoods

def inference(mock_lens_data_list, param_names_plot, param_names_macro_plot, keep_lens_index, simulation_list,
              kwargs_density, flux_ratio_measurement_uncertainty,
            imaging_data_likelihood=True,
            imaging_data_hard_cut=False,
             imaging_data_likelihood_scale=10.0,
              percentile_cut_image_data=None,
              n_resample=0, n_keep_S_statistic=None, ABC_flux_ratio_likelihood=True,
              S_statistic_tolerance=None, make_plot=False):

    _output_list = []
    pdf_list = []

    for mock_lens_index in keep_lens_index:

        mock_data_class = mock_lens_data_list[mock_lens_index - 1]
        print('number of samples in mock ' + str(mock_lens_index) + ': ',
              str(np.round(simulation_list[mock_lens_index - 1].parameters.shape[0] / 1e6, 2)) + ' million')
        kwargs_pdf = {'ABC_flux_ratio_likelihood': ABC_flux_ratio_likelihood,
                      'flux_ratio_uncertainty_percentage': [flux_ratio_measurement_uncertainty] * 3,
                      'uncertainty_in_flux_ratios': True,
                      'imaging_data_likelihood': imaging_data_likelihood,
                      'imaging_data_hard_cut': imaging_data_hard_cut,
                      'percentile_cut_image_data': percentile_cut_image_data,
                      'imaging_data_likelihood_scale': imaging_data_likelihood_scale,
                      'n_keep_S_statistic': n_keep_S_statistic,
                      'S_statistic_tolerance': S_statistic_tolerance,
                      'perturb_measurements': True
                      }
        if flux_ratio_measurement_uncertainty <= 0.001:
            n_resample = 0
        _density, _output, _ = simulation_output_to_density(deepcopy(simulation_list[mock_lens_index - 1]),
                                                            deepcopy(mock_data_class),
                                                            param_names_plot,
                                                            kwargs_pdf,
                                                            kwargs_density,
                                                            param_names_macro_plot,
                                                            n_resample=n_resample)
        _output_list.append(_output)
        pdf_list.append(_density)
        if S_statistic_tolerance is not None and n_keep_S_statistic is None:
            print('number of samples after cut on S statistic: ', len(_output.flux_ratio_summary_statistic))
    if make_plot:
        fig = plt.figure()
        fig.set_size_inches(8, 6)
        ax = plt.subplot(111)
        for _out in _output_list:
            ax.hist(_out.flux_ratio_summary_statistic, label='Lens ' + str(mock_lens_index), bins=20, range=(0.0, 0.15),
                    alpha=0.5)
        ax.legend(fontsize=14)
        ax.set_xlim(0.0, 0.15)
    return IndepdendentLikelihoods(pdf_list), _output_list

def nmax_bic_minimize(data, model_class, fitting_kwargs_list, n_max_list, verbose=True, make_plots=False):
    """

    :param data:
    :param model:
    :param fitting_kwargs_list:
    :param n_max_list:
    :param verbose:
    :return:
    """
    bic_list = []
    chain_list_list = []
    for n_max in n_max_list:
        if n_max == 0:
            model = model_class(data, shapelets_order=None)
        else:
            model = model_class(data, shapelets_order=n_max)
        kwargs_params = model.kwargs_params()
        kwargs_model, lens_model_init, kwargs_lens_init, index_lens_split = model.setup_kwargs_model()
        kwargs_constraints = model.kwargs_constraints
        kwargs_likelihood = model.kwargs_likelihood
        fitting_sequence = FittingSequence(data.kwargs_data_joint,
                                           kwargs_model,
                                           kwargs_constraints,
                                           kwargs_likelihood,
                                           kwargs_params,
                                           mpi=False, verbose=verbose)
        chain_list = fitting_sequence.fit_sequence(fitting_kwargs_list)
        bic = fitting_sequence.bic
        bic_list.append(bic)
        chain_list_list.append(chain_list)
        if make_plots:
            kwargs_result = fitting_sequence.best_fit()

            multi_band_list = data.kwargs_data_joint['multi_band_list']
            modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02,
                                  cmap_string="gist_heat",
                                  fast_caustic=True)
            for i in range(len(chain_list)):
                chain_plot.plot_chain_list(chain_list, i)
            f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
            modelPlot.data_plot(ax=axes[0, 0])
            modelPlot.model_plot(ax=axes[0, 1])
            modelPlot.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)
            modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100)
            modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
            modelPlot.magnification_plot(ax=axes[1, 2])
            f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
            modelPlot.decomposition_plot(ax=axes[0, 0], text='Lens light', lens_light_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 0], text='Lens light convolved', lens_light_add=True)
            modelPlot.decomposition_plot(ax=axes[0, 1], text='Source light', source_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 1], text='Source light convolved', source_add=True)
            modelPlot.decomposition_plot(ax=axes[0, 2], text='All components', source_add=True, lens_light_add=True,
                                         unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 2], text='All components convolved', source_add=True,
                                         lens_light_add=True, point_source_add=True)
            f.tight_layout()
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
            plt.show()

            fig = plt.figure()
            fig.set_size_inches(6, 6)
            ax = plt.subplot(111)
            kwargs_plot = {'ax': ax,
                           'index_macromodel': [0, 1],
                           'with_critical_curves': True,
                           'v_min': -0.2, 'v_max': 0.2}
            modelPlot.substructure_plot(band_index=0, **kwargs_plot)
            print(kwargs_result)
            print(kwargs_result['kwargs_lens'])
            #a = input('continue')
        print('bic: ', bic)
        print('bic list: ', bic_list)
    return bic_list, chain_list_list

def cut_on_data(output, data,
                ABC_flux_ratio_likelihood=True,
                flux_uncertainty_percentage=None,
                flux_ratio_uncertainty_percentage=None,
                uncertainty_in_flux_ratios=True,
                imaging_data_likelihood=True,
                imaging_data_hard_cut=False,
                percentile_cut_image_data=None,
                n_keep_S_statistic=None,
                S_statistic_tolerance=None,
                perturb_measurements=True,
                perturb_model=True,
                imaging_data_likelihood_scale=20,
                cut_image_data_first=True,
                verbose=False):
    """

    :param output:
    :param data:
    :param ABC_flux_ratio_likelihood:
    :param flux_uncertainty_percentage:
    :param flux_ratio_uncertainty_percentage:
    :param imaging_data_likelihood:
    :param imaging_data_hard_cut:
    :param percentile_cut_image_data:
    :param n_keep_S_statistic:
    :param S_statistic_tolerance:
    :param perturb_measurements:
    :return:
    """
    data_class = deepcopy(data)
    __out = deepcopy(output)

    if imaging_data_hard_cut is False:
        percentile_cut_image_data = 100.0 # keep everything
    else:
        assert percentile_cut_image_data is not None

    if uncertainty_in_flux_ratios:
        mags_measured = data_class.magnifications
        _flux_ratios_measured = mags_measured[1:] / mags_measured[0]
        if flux_ratio_uncertainty_percentage is None:
            flux_ratios_measured = data_class.magnifications[1:] / data_class.magnifications[0]
        elif perturb_measurements:
            delta_f = np.array(flux_ratio_uncertainty_percentage) * np.array(_flux_ratios_measured)
            flux_ratios_measured = [np.random.normal(_flux_ratios_measured[i], delta_f[i]) for i in range(0, 3)]
        else:
            flux_ratios_measured = _flux_ratios_measured
        if ABC_flux_ratio_likelihood:
            if perturb_model:
                if flux_ratio_uncertainty_percentage is None:
                    model_flux_ratios = __out.flux_ratios
                else:
                    model_flux_ratios = perturbed_flux_ratios_from_flux_ratios(__out.flux_ratios,
                                                                           flux_ratio_uncertainty_percentage)
            else:
                model_flux_ratios = __out.flux_ratios
            __out.set_flux_ratio_summary_statistic(None,
                                                   None,
                                                   measured_flux_ratios=flux_ratios_measured,
                                                   modeled_flux_ratios=model_flux_ratios,
                                                   verbose=verbose)

        else:
            model_flux_ratios = __out.flux_ratios
            __out.set_flux_ratio_likelihood(None,
                                            None,
                                            flux_ratio_uncertainty_percentage,
                                            measured_flux_ratios=flux_ratios_measured,
                                            modeled_flux_ratios=model_flux_ratios,
                                            verbose=verbose)

    else:
        if flux_uncertainty_percentage is None:
            model_image_magnifications = __out.image_magnifications
        else:
            if perturb_measurements:
                data_class.perturb_flux_measurements(flux_uncertainty_percentage)
            model_image_magnifications = perturbed_fluxes_from_fluxes(__out.image_magnifications,
                                                                      flux_uncertainty_percentage)

        observed_image_magnifications = data_class.magnifications
        if ABC_flux_ratio_likelihood:
            __out.set_flux_ratio_summary_statistic(observed_image_magnifications,
                                                 model_image_magnifications)
        else:
            __out.set_flux_ratio_likelihood(observed_image_magnifications,
                                          model_image_magnifications,
                                          flux_ratio_uncertainty_percentage)

    if cut_image_data_first:
        _out = __out.cut_on_image_data(percentile_cut=percentile_cut_image_data)
        if ABC_flux_ratio_likelihood:
            # now cut on flux ratios
            if S_statistic_tolerance is not None:
                assert n_keep_S_statistic is None
                n_keep_S_statistic = np.sum(_out.flux_ratio_summary_statistic < S_statistic_tolerance)
            weights_flux_ratios = 1.0
            out_cut_S = _out.cut_on_S_statistic(keep_best_N=n_keep_S_statistic)
        else:
            n_keep_S_statistic = -1
            out_cut_S = _out.cut_on_S_statistic(keep_best_N=n_keep_S_statistic)
            weights_flux_ratios = out_cut_S.flux_ratio_likelihood

        if imaging_data_likelihood:
            assert imaging_data_hard_cut is False
            relative_log_likelihoods = out_cut_S.image_data_logL - np.max(out_cut_S.image_data_logL)
            rescale_log_like = 1.0
            weights_imaging_data = np.exp(relative_log_likelihoods / rescale_log_like)
            effective_sample_size = np.sum(weights_imaging_data)
            target_sample_size = len(weights_imaging_data) / imaging_data_likelihood_scale
            while effective_sample_size < target_sample_size:
                rescale_log_like += 1
                weights_imaging_data = np.exp(relative_log_likelihoods / rescale_log_like)
                effective_sample_size = np.sum(weights_imaging_data)
                target_sample_size = len(weights_imaging_data) / imaging_data_likelihood_scale
            #print('rescaled relative log-likelihoods by '+str(rescale_log_like))
        else:
            weights_imaging_data = np.ones(out_cut_S.parameters.shape[0])
    else:
        if ABC_flux_ratio_likelihood:
            # now cut on flux ratios
            if S_statistic_tolerance is not None:
                assert n_keep_S_statistic is None
                n_keep_S_statistic = np.sum(__out.flux_ratio_summary_statistic < S_statistic_tolerance)
            weights_flux_ratios = 1.0
            _out = __out.cut_on_S_statistic(keep_best_N=n_keep_S_statistic)
        else:
            n_keep_S_statistic = -1
            _out = __out.cut_on_S_statistic(keep_best_N=n_keep_S_statistic)
            weights_flux_ratios = _out.flux_ratio_likelihood

        if imaging_data_likelihood:
            assert imaging_data_hard_cut is False
            relative_log_likelihoods = _out.image_data_logL - np.max(_out.image_data_logL)
            rescale_log_like = 1.0
            weights_imaging_data = np.exp(relative_log_likelihoods / rescale_log_like)
            effective_sample_size = np.sum(weights_imaging_data)
            target_sample_size = len(weights_imaging_data) / imaging_data_likelihood_scale
            while effective_sample_size < target_sample_size:
                rescale_log_like += 1
                weights_imaging_data = np.exp(relative_log_likelihoods / rescale_log_like)
                effective_sample_size = np.sum(weights_imaging_data)
                target_sample_size = len(weights_imaging_data) / imaging_data_likelihood_scale
        else:
            weights_imaging_data = np.ones(_out.parameters.shape[0])
        out_cut_S = _out.cut_on_image_data(percentile_cut=percentile_cut_image_data)

    return out_cut_S, weights_imaging_data * weights_flux_ratios

def simulation_output_to_density(output, data, param_names_plot, kwargs_cut_on_data, kwargs_density,
                                 param_names_macro_plot=None, n_resample=0, custom_weights=None, apply_cuts=True):

    if apply_cuts:
        out, weights = cut_on_data(output, data, **kwargs_cut_on_data)
        if custom_weights is not None:
            for single_weights in custom_weights:
                (param, mean, sigma) = single_weights
                weights *= np.exp(-0.5 * (out.param_dict[param] - mean) **2 / sigma**2)
        for i in range(0, n_resample):
            _out, _weights = cut_on_data(output, data, **kwargs_cut_on_data)
            out = Output.join(out, _out)
            if custom_weights is not None:
                for single_weights in custom_weights:
                    (param, mean, sigma) = single_weights
                    _weights *= np.exp(-0.5 * (_out.param_dict[param] - mean) ** 2 / sigma ** 2)
            weights = np.append(weights, _weights)
        weights = [weights]
    else:
        out = output
        weights = None
    samples = None
    if len(param_names_plot) > 0:
        samples = out.parameter_array(param_names_plot)
    if param_names_macro_plot is not None:
        samples_macro = out.macromodel_parameter_array(param_names_macro_plot)
        if samples is None:
            samples = samples_macro
        else:
            samples = np.hstack((samples, samples_macro))
        param_names = param_names_plot + param_names_macro_plot
    else:
        param_names = param_names_plot
    from trikde.pdfs import DensitySamples
    density = DensitySamples(samples, param_names, weights, **kwargs_density)
    return density, out, weights

# def likelihood_function_change(pdf1, pdf2):
#
#     from trikde.

