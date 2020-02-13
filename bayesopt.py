# @author: Robin Ru (robin@robots.ox.ac.uk)

import pickle
import time

import numpy as np

from acq_funcs.acq_optimizer import Acq_Optimizer
from acq_funcs.acquisitions import LCB_budget, LCB_budget_additive
from models.additive_gp_decomp import Additive_GPModel_Learn_Decomp
from models.gp import GPModel
from models.gpdr import GPModelLDR
from utilities.upsampler import upsample_projection


class Bayes_opt():
    def __init__(self, func, bounds, saving_path):
        """
        Bayesian Optimisation algorithm

        :param func: the objective function to be optimised
        :param bounds: the input space bounds
        :param saving_path: saving path for failed BO runs (rarely occurred)
        """

        self.func = func
        self.bounds = bounds
        self.noise_var = 1.0e-10
        self.saving_path = saving_path

    def initialise(self, X_init=None, Y_init=None, model_type='GP',
                   acq_type='LCB', batch_option='CL', batch_size=1, sparse=None, seed=42, nchannel=3,
                   high_dim=int(32 * 32),
                   ARD=False, cost_metric=None, nsubspaces=1, normalize_Y=True, update_freq=10, dim_reduction='BILI'):
        """
        :param X_init: initial observation input data
        :param Y_init: initial observation input data
        :param model_type: BO surrogate model type
        :param acq_type: BO acquisition function type
        :param batch_option: for selecting a batch of new locations to be evaluated in parallel next
        :param batch_size: the number of new query locations in the batch (=1 for sequential BO and > 1 for parallel BO)
        :param sparse: sparse GP options
        :param seed: random seed
        :param nchannel: number of image channels
        :param high_dim: image dimension (e.g. 32x32 for CIFAR10) or high-dimensional search space for imagenet (96x96)
        :param ARD: ARD option for GP models
        :param cost_metric: perturbatino cost metric; if None, the acqusition equals to normal LCB acquisition function
        :param nsubspaces: number of subspaces in the decomposition for ADDGP only
        :param normalize_Y: normalise output data
        :param update_freq: frequency of relearning GP hyperparameters
        :param dim_reduction: dimension reduction method used in upsampling
        """
        assert X_init.ndim == 2, "X_init has to be 2D array"
        assert Y_init.ndim == 2, "Y_init has to be 2D array"

        self.X_init = X_init
        self.Y_init = Y_init
        self.X = np.copy(X_init)
        self.Y = np.copy(Y_init)
        self.acq_type = acq_type
        self.batch_option = batch_option
        self.batch_size = batch_size
        self.model_type = model_type
        self.seed = seed
        self.normalize_Y = normalize_Y
        self.nchannel = nchannel
        self.X_dim = self.X.shape[1]
        self.high_dim = high_dim
        self.dim_reduction = dim_reduction

        # Find the minimum observed functional value and its location
        self.arg_opt = np.atleast_2d(self.X[np.argmin(self.Y)])
        self.minY = np.min(self.Y)

        # Choose the surrogate model for BO
        if model_type == 'GP':
            nsubspaces = 1
            if self.noise_var > 1e-6:
                self.model = GPModel(noise_var=self.noise_var, ARD=ARD, seed=seed,
                                     normalize_Y=self.normalize_Y,
                                     update_freq=update_freq, sparse=sparse)
            else:
                self.model = GPModel(exact_feval=True, ARD=ARD, seed=seed, normalize_Y=self.normalize_Y,
                                     update_freq=update_freq, sparse=sparse)

        if model_type == 'GPLDR':
            nsubspaces = 1
            if self.noise_var > 1e-6:
                self.model = GPModelLDR(noise_var=self.noise_var, ARD=ARD, seed=seed, high_dim=high_dim,
                                        dim_reduction=dim_reduction, sparse=sparse,
                                        normalize_Y=self.normalize_Y, update_freq=update_freq, nchannel=nchannel)
            else:
                self.model = GPModelLDR(exact_feval=True, ARD=ARD, seed=seed, high_dim=high_dim,
                                        dim_reduction=dim_reduction, sparse=sparse,
                                        normalize_Y=self.normalize_Y, update_freq=update_freq, nchannel=nchannel)

        elif model_type == 'ADDGPLD':
            print(f'nsubspaces={nsubspaces}')
            if self.noise_var > 1e-6:
                self.model = Additive_GPModel_Learn_Decomp(noise_var=self.noise_var, ARD=ARD,
                                                           sparse=sparse,
                                                           seed=seed, normalize_Y=self.normalize_Y,
                                                           n_subspaces=nsubspaces, update_freq=update_freq)
            else:
                self.model = Additive_GPModel_Learn_Decomp(exact_feval=True, ARD=ARD, sparse=sparse,
                                                           seed=seed, normalize_Y=self.normalize_Y,
                                                           n_subspaces=nsubspaces, update_freq=update_freq)

        # Choose the acquisition function for BO
        if self.acq_type == 'LCB':
            if cost_metric == 10:
                cost_metric = np.inf
            if model_type.startswith('ADDGP'):
                acqu_func = LCB_budget_additive(self.model, dis_metric=cost_metric)
            else:
                acqu_func = LCB_budget(self.model, dis_metric=cost_metric)
        else:
            print('Not implemented')
        self.query_strategy = Acq_Optimizer(model=self.model, acqu_func=acqu_func, bounds=self.bounds,
                                            batch_size=batch_size, batch_method=batch_option, model_name=model_type,
                                            nsubspace=nsubspaces)

    def run(self, total_iterations):
        """
        :param total_iterations:
        # X_query, Y_query - query points selected by BO;
            # X_opt, Yopt      - guesses of the global optimum/optimiser (= optimum point of GP posterior mean)
        :return X_query: inputs queried by BO;
        :return Y_query: output values at queried locations
        :return X_opt: the guess of the global optimum location
        :return Yopt: the guess of the global optimum value
        :return time_record: BO time array for all iterations
        """

        np.random.seed(self.seed)
        X_query = np.copy(self.X)
        Y_query = np.copy(self.Y)
        X_opt = np.copy(np.atleast_2d(self.arg_opt))
        Y_opt = np.copy(np.atleast_2d(self.minY))
        time_record = np.zeros([total_iterations, 2])
        self.e = np.exp(1)
        self.opt_dr_list = []

        # Upsample the observed data to image dimension in the case of auto-learning of d^r
        if 'LDR' in self.model_type:
            x_curr_dim = self.X.shape[1]
            if int(x_curr_dim / self.nchannel) < self.high_dim:
                self.X = upsample_projection(self.dim_reduction, X_query, low_dim=int(x_curr_dim / self.nchannel),
                                             high_dim=self.high_dim, nchannel=self.nchannel)

        # Fit GP model to the observed data
        self.model._update_model(self.X, self.Y, itr=0)

        for k in range(total_iterations):

            # Optimise the acquisition function to get the next query point and evaluate at next query point
            start_time_opt = time.time()
            x_next_batch, acqu_value_batch = self.query_strategy.get_next(self.X, self.Y)
            max_acqu_value = np.max(acqu_value_batch)
            t_opt_acq = time.time() - start_time_opt
            time_record[k, 0] = t_opt_acq

            # Upsample the observed data to image dimension in the case of auto-learning of d^r after each iteration
            if 'LDR' in self.model_type:
                self.opt_dr_list.append(self.model.opt_dr)
                x_curr_dim = x_next_batch.shape[1]
                if int(x_curr_dim / self.nchannel) < self.high_dim:
                    x_next_batch = upsample_projection(self.dim_reduction, x_next_batch,
                                                       low_dim=int(x_curr_dim / self.nchannel), high_dim=self.high_dim,
                                                       nchannel=self.nchannel)
            else:
                self.opt_dr_list.append(np.atleast_2d(0))

            # Evaluate the objective function at the next query point
            y_next_batch = self.func(x_next_batch) + np.random.normal(0, np.sqrt(self.noise_var),
                                                                      (x_next_batch.shape[0], 1))
            # Augment the observed data
            self.X = np.vstack((self.X, x_next_batch))
            self.Y = np.vstack((self.Y, y_next_batch))
            self.minY = np.min(self.Y)

            #  Store the intermediate BO results
            X_query = np.vstack((X_query, np.atleast_2d(x_next_batch)))
            Y_query = np.vstack((Y_query, np.atleast_2d(y_next_batch)))
            X_opt = np.concatenate((X_opt, np.atleast_2d(X_query[np.argmin(Y_query), :])))
            Y_opt = np.concatenate((Y_opt, np.atleast_2d(min(Y_query))))

            print(f'{self.model_type}{self.acq_type}{self.batch_option} ||'
                  f'seed:{self.seed},itr:{k}, y_next:{np.min(y_next_batch)}, y_opt:{Y_opt[-1, :]}')

            # Terminate the BO loop if the attack succeeds
            if min(Y_query) <= 0:
                break

            # Update the surrogate model with new data
            start_time_update = time.time()
            try:
                self.model._update_model(self.X, self.Y, itr=k)
            except:
                # If the model update fails, terminate the BO loop
                partial_results = {'X_query': self.X.astype(np.float16),
                                   'Y_query': self.Y.astype(np.float16),
                                   'model_kernel': self.model.model.kern}
                failed_file_name = self.saving_path
                with open(failed_file_name, 'wb') as file:
                    pickle.dump(partial_results, file)
                print('This BO target failed')
                assert False
            t_update_model = time.time() - start_time_update
            time_record[k, 1] = t_update_model
            print(f'Time for optimising acquisition function={t_opt_acq}; '
                  f'Time for updating the model={t_update_model}')

        return X_query, Y_query, X_opt, Y_opt, time_record
