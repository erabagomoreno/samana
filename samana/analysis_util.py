import numpy as np
from lenstronomy.Workflow.fitting_sequence import FittingSequence

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
