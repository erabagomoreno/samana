import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from samana.analysis_util import simulation_output_to_density
from trikde.pdfs import IndepdendentLikelihoods

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.major.width'] = 3.5
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.major.width'] = 3.5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20

def setup_macromodel_inference_plot(output, mock_data_class,
                                    param_names_plot_macro=None,
                                    param_names_plot=None,
                                    flux_ratio_uncertainty=0.001,
                                    imaging_data_hard_cut=True,
                                    imaging_data_likelihood=False,
                                    percentile_cut_image_data=10,
                                    n_keep=2000,
                                    s_cut=None,
                                    imaging_data_likelihood_scale=20,
                                    nbins=10,
                                    use_kde=False,
                                    param_ranges=None,
                                    n_resample=5):

    if flux_ratio_uncertainty <= 0.001:
        n_resample = 0
    if param_names_plot is None:
        param_names_plot = []
    if param_names_plot_macro is None:
        param_names_plot_macro = ['theta_E', 'q', 'gamma_ext', 'gamma', 'a4_a_cos', 'a3_a_cos']
    kwargs_pdf = {'ABC_flux_ratio_likelihood': True,
                  'flux_ratio_uncertainty_percentage': [flux_ratio_uncertainty] * 3,
                  'uncertainty_in_flux_ratios': True,
                  'imaging_data_likelihood': imaging_data_likelihood,
                  'imaging_data_hard_cut': imaging_data_hard_cut,
                  'percentile_cut_image_data': percentile_cut_image_data,
                  'n_keep_S_statistic': n_keep,
                  'S_statistic_tolerance': s_cut,
                  'perturb_measurements': True,
                  'imaging_data_likelihood_scale': imaging_data_likelihood_scale
                  }
    kwargs_density = {'nbins': nbins,
                      'use_kde': use_kde,
                      'param_ranges': param_ranges}
    density, _, _ = simulation_output_to_density(deepcopy(output),
                                                 deepcopy(mock_data_class),
                                                 param_names_plot,
                                                 kwargs_pdf,
                                                 kwargs_density,
                                                 param_names_plot_macro,
                                                 n_resample=n_resample)
    like = IndepdendentLikelihoods([density])
    param_ranges_density = density.param_ranges
    return like, param_ranges_density

def mock_lens_data_plot(image_sim, window_size, vminmax=1.5, cmap='gist_heat', label='', zd='', zs='',
                        save_fig=False, filename=None):
    fig = plt.figure(1)
    fig.set_size_inches(8, 8)
    ax = plt.subplot(111)
    extent = [-window_size / 2, window_size / 2, -window_size / 2, window_size / 2]
    image = deepcopy(image_sim)
    image[np.where(image < 10 ** -vminmax)] = 10 ** -vminmax
    im = ax.imshow(np.log10(image), origin='lower', alpha=0.75,
                   vmin=-vminmax, vmax=vminmax, cmap=cmap, extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    xlow = -window_size / 2 + 0.2
    xhigh = xlow + 1.0
    y = -window_size / 2 + 0.3
    ax.plot([xlow, xhigh], [y, y], color='w', lw=4)
    ax.annotate('1 arcsec', xy=(xlow + 0.165, y - 0.21), fontsize=18, color='w')
    ax.annotate(label + '\n' + r'$z_{\rm{d}} = $' + str(zd) + '\n' + r'$z_{\rm{s}} = $' + str(zs), xy=(0.035, 0.78),
                xycoords='axes fraction', fontsize=24, color='w')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_{10} \ \rm{intensity \ [counts/pixel]}$', fontsize=25)
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
    plt.show()

def mock_substructure_plot(x_image, y_image, window_size, n_pixels, lens_model, lens_model_macro, kwargs_lens,
                           kwargs_lens_macro, label='', save_fig=False, filename=None):

    _r = np.linspace(-window_size / 2, window_size / 2, n_pixels)
    _xx, _yy = np.meshgrid(_r, _r)
    shape0 = _xx.shape
    kappa_macro = lens_model_macro.kappa(_xx.ravel(), _yy.ravel(), kwargs_lens_macro).reshape(shape0)
    kappa = lens_model.kappa(_xx.ravel(), _yy.ravel(), kwargs_lens).reshape(shape0)
    residual_kappa = kappa - kappa_macro
    mean = np.mean(residual_kappa[np.where(np.hypot(_xx, _yy) > 0.2)])

    fig = plt.figure(1)
    fig.set_size_inches(8, 8)
    ax = plt.subplot(111)
    extent = [-window_size / 2, window_size / 2, -window_size / 2, window_size / 2]

    im = ax.imshow(residual_kappa - mean, origin='lower', vmin=-0.1, vmax=0.1, cmap='seismic', alpha=1.,
                   extent=extent)
    ax.scatter(x_image, y_image, color='k', marker='x', s=200, alpha=0.8)
    ax.scatter(x_image, y_image, color='k', marker='+', s=250, alpha=0.8)

    ext = LensModelExtensions(lens_model)
    ra_crit_list, dec_crit_list, _, _ = ext.critical_curve_caustics(kwargs_lens, compute_window=window_size,
                                                                    grid_scale=0.01)
    for (racrit, deccrit) in zip(ra_crit_list[0:1], dec_crit_list[0:1]):
        ax.plot(racrit, deccrit, color='k')

    ax.set_xticks([])
    ax.set_yticks([])
    xlow = -window_size / 2 + 0.2
    xhigh = xlow + 1.0
    y = -window_size / 2 + 0.3
    ax.plot([xlow, xhigh], [y, y], color='k', lw=4)
    ax.annotate('1 arcsec', xy=(xlow + 0.165, y - 0.21), fontsize=18, color='k')
    ax.annotate(label, xy=(0.035, 0.9),
                xycoords='axes fraction', fontsize=24, color='k')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[-0.1, -0.05, 0.0, 0.05, 0.1])
    cbar.set_label('DM substructure convergence', fontsize=20)
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
    plt.show()
