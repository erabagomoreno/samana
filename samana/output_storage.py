import numpy as np

class Output(object):

    def __init__(self, parameters, image_magnifications, macromodel_samples, fitting_kwargs_list=None,
                 param_names=None, macromodel_sample_names=None):
        """

        :param param_names:
        :param parameters:
        :param image_magnifications:
        :param macromodel_sample_names:
        :param macromodel_samples:
        :param fitting_kwargs_list:
        """
        self.parameters = parameters
        self.image_magnifications = image_magnifications
        self.macromodel_samples = macromodel_samples
        self.fitting_kwargs_list = fitting_kwargs_list
        self.seed = parameters[:, -1]
        self.image_data_logL = parameters[:, -2]
        self.flux_ratio_likelihood = parameters[:, -3]
        self.flux_ratio_stat = parameters[:, -4]
        self._param_dict = None
        self._param_names = param_names
        self._macromodel_sample_names = macromodel_sample_names
        if param_names is not None:
            assert len(param_names) == parameters.shape[1]
            self._param_dict = {}
            for i, name in enumerate(param_names):
                self._param_dict[name] = parameters[:, i]
        self._macromodel_samples_dict = None
        if macromodel_sample_names is not None:
            assert len(macromodel_sample_names) == macromodel_samples.shape[1]
            self._macromodel_samples_dict = {}
            for i, name in enumerate(macromodel_sample_names):
                self._macromodel_samples_dict[name] = macromodel_samples[:, i]

    @classmethod
    def from_raw_output(cls, output_path, job_index_min, job_index_max, fitting_kwargs_list=None):

        param_names = None
        macro_param_names = None
        init = True
        for i in range(job_index_min, job_index_max+1):

            folder = output_path + '/job_'+str(i)+'/'
            try:
                params = np.loadtxt(folder + 'parameters.txt', skiprows=1)
            except:
                print('params file '+folder+'parameters.txt not found... ')
                continue
            try:
                fluxes = np.loadtxt(folder + 'fluxes.txt')
            except:
                print('fluxes file ' + folder + 'fluxes.txt not found... ')
                continue
            try:
                macrosamples = np.loadtxt(folder + 'macromodel_samples.txt', skiprows=1)
            except:
                print('macromodel samples file ' + folder + 'macromodel_samples.txt not found... ')
                continue
            # check the arrays are all the same length
            size_params = params.shape[0]
            size_fluxes = fluxes.shape[0]
            size_macro = macrosamples.shape[0]
            if size_params != size_fluxes:
                print('parameters and fluxes have different shape for '+folder)
                continue
            if size_params != size_macro:
                print('parameters and macromodel samples have different shape for '+folder)
                continue
            if param_names is None:
                with open(folder + 'parameters.txt', 'r') as f:
                    param_names = f.readlines(1)[0].split()
                f.close()
            if macro_param_names is None:
                with open(folder + 'macromodel_samples.txt', 'r') as f:
                    macromodel_sample_names = f.readlines(1)[0].split()
                f.close()
            if init:
                parameters = params
                magnifications = fluxes
                macromodel_samples = macrosamples
                init = False
            else:
                parameters = np.vstack((parameters, params))
                magnifications = np.vstack((magnifications, fluxes))
                macromodel_samples = np.vstack((macromodel_samples, macrosamples))

        return Output(parameters, magnifications, macromodel_samples, fitting_kwargs_list,
                 param_names, macromodel_sample_names)

    @property
    def param_dict(self):

        if self._param_dict is None:
            if self._param_names is not None:
                assert len(self._param_names) == self.parameters.shape[1]
                self._param_dict = {}
                for i, name in enumerate(self._param_names):
                    self._param_dict[name] = self.parameters[:, i]
            else:
                print('parameter names need to be specified to create a param dictionary')
                return None
        else:
            return self._param_dict

    @property
    def macromodel_samples_dict(self):
        
        if self._macromodel_samples_dict is None:
            if self._macromodel_sample_names is not None:
                assert len(self._macromodel_sample_names) == self.macromodel_samples.shape[1]
                self._macromodel_samples_dict = {}
                for i, name in enumerate(self._macromodel_sample_names):
                    self._macromodel_samples_dict[name] = self.macromodel_samples[:, i]
            else:
                print('parameter names need to be specified to create a param dictionary')
                return None
        else:
            return self._macromodel_samples_dict

    def _subsample(self, inds_keep):
        """
        :param inds_keep:
        :return:
        """
        parameters = self.parameters[inds_keep, :]
        image_magnifications = self.image_magnifications[inds_keep, :]
        macromodel_samples = self.macromodel_samples[inds_keep, :]
        return Output(parameters, image_magnifications, macromodel_samples,
                      fitting_kwargs_list=self.fitting_kwargs_list,
                      param_names=self._param_names,
                      macromodel_sample_names=self._macromodel_sample_names)

    def cut_on_image_data(self, percentile_cut):
        """

        :param percentile_cut:
        :return:
        """
        inds_sorted = np.argsort(self.image_data_logL)
        idx_cut = int((100 - percentile_cut) / 100 * len(self.image_data_logL))
        logL_cut = self.image_data_logL[inds_sorted[idx_cut]]
        inds_keep = np.where(self.image_data_logL >= logL_cut)[0]
        return self._subsample(inds_keep)

    def cut_on_S_statistic(self, keep_best_N=None, percentile_cut=None, S_statistic_cut=None):
        """

        :param percentile_cut:
        :return:
        """
        sorted_inds = np.argsort(self.flux_ratio_stat)
        if keep_best_N is not None:
            assert percentile_cut is None and S_statistic_cut is None
            inds_keep = sorted_inds[0:keep_best_N]
        elif percentile_cut is not None:
            assert S_statistic_cut is None
            idxcut = int(self.parameters.shape[0] * percentile_cut/100)
            inds_keep = sorted_inds[0:idxcut]
        elif S_statistic_cut is not None:
            inds_keep = np.where(self.flux_ratio_stat <= S_statistic_cut)[0]
        else:
            raise Exception('must specify keep_best_N, percentile_cut, or S_statistic_cut')
        return self._subsample(inds_keep)

    def cut_on_flux_ratio_likelihood(self, keep_best_N=None, percentile_cut=None, likelihood_cut=None):
        """

        :param percentile_cut:
        :return:
        """
        sorted_inds = np.argsort(self.flux_ratio_likelihood)
        if keep_best_N is not None:
            assert percentile_cut is None and likelihood_cut is None
            inds_keep = sorted_inds[0:keep_best_N]
        elif percentile_cut is not None:
            assert likelihood_cut is None
            idxcut = int(self.parameters.shape[0] * percentile_cut/100)
            inds_keep = sorted_inds[0:idxcut]
        elif likelihood_cut is not None:
            inds_keep = np.where(self.flux_ratio_likelihood <= likelihood_cut)[0]
        else:
            raise Exception('must specify keep_best_N, percentile_cut, or likelihood_cut')
        return self._subsample(inds_keep)
