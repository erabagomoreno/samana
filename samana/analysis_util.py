import numpy as np
from copy import deepcopy
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from samana.image_magnification_util import perturbed_fluxes_from_fluxes, perturbed_flux_ratios_from_flux_ratios
from samana.output_storage import Output

def nmax_bic_minimize(data, model_class, fitting_kwargs_list, n_max_list, verbose=True):
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
            model = model_class(data)
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
        weights_imaging_data = np.exp(out_cut_S.image_data_logL - np.max(out_cut_S.image_data_logL))
    else:
        weights_imaging_data = np.ones(out_cut_S.parameters.shape[0])

    return out_cut_S, weights_imaging_data * weights_flux_ratios

def simulation_output_to_density(output, data, param_names_plot, kwargs_cut_on_data, kwargs_density,
                                 param_names_macro_plot=None, n_resample=0):

    out, weights = cut_on_data(output, data, **kwargs_cut_on_data)
    for i in range(0, n_resample):
        _out, _weights = cut_on_data(output, data, **kwargs_cut_on_data)
        out = Output.join(out, _out)
        weights = np.append(weights, _weights)

    samples = out.parameter_array(param_names_plot)
    if param_names_macro_plot is not None:
        samples_macro = out.macromodel_parameter_array(param_names_macro_plot)
        samples = np.hstack((samples, samples_macro))
        param_names = param_names_plot + param_names_macro_plot
    else:
        param_names = param_names_plot
    from trikde.pdfs import DensitySamples
    density = DensitySamples(samples, param_names, [weights], **kwargs_density)
    return density, out, weights
