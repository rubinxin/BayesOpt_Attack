# @author: Robin Ru (robin@robots.ox.ac.uk)

import GPy
import numpy as np

from utilities.utilities import subset_select, subset_select_for_learning
from .base import BaseModel


def split(a, n):
    # split array a into n approximately equal splits
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


class Additive_GPModel_Learn_Decomp(BaseModel):

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs',
                 max_iters=1000, optimize_restarts=5, sparse=None, num_inducing=10,
                 verbose=False, ARD=False, seed=42, normalize_Y=True, n_subspaces=12, update_freq=5):
        """
        Additive GP model which learns the decomposition.

        :param kernel: the GP kernel option; use default if set to None
        :param noise_var: observation noise variance
        :param exact_feval: whether to learn observation noise (exact_feval=False) or not (exact_feval=True)
        :param optimizer: to optimise marginal likelihood w.r.t GP hyperparameters
        :param max_iters: max iterations for optimising marginal likelihood
        :param optimize_restarts: number of restart for optimising marginal likelihood
        :param sparse:
        :param num_inducing: number of inducing points if a sparse GP is used.
        :param verbose: enable print-statements
        :param ARD: whether ARD is used in the kernel (default, False).
        :param seed: set ranfom seed
        :param normalize_Y: normalise output data
        :param n_subspaces: number of subspaces to be decomposed into
        :param update_freq: frequency of relearning GP hyperparameters
        """
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse = sparse
        self.sparse_surrogate = False
        self.num_inducing = num_inducing
        self.model = None
        self.ARD = ARD
        self.seed = seed
        self.update_interval = update_freq
        self.normalize_Y = normalize_Y
        self.n_sub = n_subspaces
        self.n_decomp = 20
        np.random.rand(self.seed)

    def _create_model_sub(self, X, Y, input_dim_permutate_list):
        """
        Creates the model for a subspace of dimensions

        :param X: observed input data
        :param Y: observed output data
        :param input_dim_permutate_list: shuffled input dimension list
        """
        # split the input dimensions into nsubspaces
        subspace_dim_list = split(input_dim_permutate_list, self.n_sub)
        # define the additive kernel
        for i in range(self.n_sub):
            # get active dimensions for each subspace_id
            subspace_dim = subspace_dim_list[i]
            kern_i = GPy.kern.Matern52(len(subspace_dim), variance=1., ARD=self.ARD,
                                       active_dims=subspace_dim, name=f'k{i}')
            # kern_i.variance.fix()
            if i == 0:
                kern = kern_i
            else:
                kern += kern_i

        # define GP model
        noise_var = Y.var() * 0.01 if self.noise_var is None else self.noise_var
        if not self.sparse_surrogate:
            sub_model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        else:
            sub_model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing)

        if self.exact_feval:
            # restrict noise variance if exact evaluations of the objective
            sub_model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # bound the noise variance if not
            sub_model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False)

        # optimise the GP hyperparameters
        try:
            sub_model.optimize(optimizer=self.optimizer, max_iters=self.max_iters, messages=False,
                               ipython_notebook=False)
            model_log_likelihood = sub_model.log_likelihood()
        except:
            model_log_likelihood = -100.00

        return sub_model, model_log_likelihood

    def _update_model(self, X_all, Y_all_raw, itr=0):
        """
        :param X_all: observed input data
        :param Y_all_raw: observed output raw data
        :param itr: BO iteration counter
        """
        # normalise the observed output raw data
        if self.normalize_Y:
            Y_all = (Y_all_raw - Y_all_raw.mean()) / (Y_all_raw.std())
        else:
            Y_all = Y_all_raw

        # if use sparse GP option, select only a subset of observed data for training surrogate (one-type of sparse GP)
        if self.sparse.startswith('SUB'):
            X_all, Y_all = subset_select(X_all, Y_all, select_metric=self.sparse)

        # initialise model or update the observed data for the model
        if self.model is None:
            self.input_dim = X_all.shape[1]
            self.input_dim_opt_ex = list(range(self.input_dim))
        else:
            self.model.set_XY(X_all, Y_all)

        if itr % int(self.update_interval * 8) == 0:
            # learn decomposition at the interval of (8 * interval of relearning hyperparameter)
            input_dim_permutate_list = [np.random.permutation(range(self.input_dim)) for i in range(self.n_decomp)]
            input_dim_permutate_list.append(self.input_dim_opt_ex)

            if 'ADD' in self.sparse:
                print('learn the decomposition with subset observed data')
                X_ob, Y_ob = subset_select_for_learning(X_all, Y_all, select_metric=self.sparse)
            else:
                X_ob, Y_ob = X_all, Y_all

            ll_list = []
            submodel_list = []
            for i, input_dim_i in enumerate(input_dim_permutate_list):
                sub_model_i, ll_i = self._create_model_sub(X_ob, Y_ob, input_dim_i)
                print(f'll for decom {i} ={ll_i}')
                ll_list.append(ll_i)
                submodel_list.append(sub_model_i)

            # select the decomposition with the maximum marginal likelihood
            mlh_idx = np.argmax(ll_list)
            self.model = submodel_list[mlh_idx]
            self.model.set_XY(X_all, Y_all)

            # set to the selected decomposition
            input_dim_opt = input_dim_permutate_list[mlh_idx]
            self.active_dims_list = split(input_dim_opt, self.n_sub)
            self.model_kern_list = [self.model.kern.__dict__[f'k{k_indx}'] for k_indx in range(self.n_sub)]
            self.input_dim_opt_ex = input_dim_opt.copy()

            print(f'opt_decom={mlh_idx}')

        if itr % self.update_interval == 0:
            # relearn the GP hyperparameters at a certain iteration interval
            self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer,
                                         max_iters=self.max_iters, verbose=self.verbose)

    def predictSub(self, X, subspace_id=0):
        """
        :param X: test location
        :param subspace_id: decide the specific subspace of active dimensions
        :return m_sub: predictive posterior mean in this subspace
        :return s_sub: predictive posterior standard deviation in this subspace
        """

        if X.ndim == 1: X = X[None, :]
        m_sub, v_sub = self.model.predict(X, kern=self.model_kern_list[subspace_id])
        v_sub = np.clip(v_sub, 1e-10, np.inf)

        return m_sub, np.sqrt(v_sub)

    def predictSub_withGradients(self, X, subspace_id=0):
        """
        :param X: test location
        :param subspace_id: decide the specific subspace of active dimensions
        :return m_sub: predictive posterior mean in this subspace
        :return s_sub: predictive posterior standard deviation in this subspace
        :return dmdx_sub: derivative of predictive posterior mean in this subspace
        :return dsdx_sub: derivative of predictive posterior standard deviation in this subspace
        """

        if X.ndim == 1: X = X[None, :]
        m_sub, v_sub = self.model.predict(X, kern=self.model_kern_list[subspace_id])
        v_sub = np.clip(v_sub, 1e-10, np.inf)

        dmdx, dvdx = self.model.predictive_gradients(X, kern=self.model_kern_list[subspace_id])
        dmdx = dmdx[:, :, 0]
        dsdx = dvdx / (2 * np.sqrt(v_sub))

        dmdx_sub = dmdx[:, self.active_dims_list[subspace_id]]
        dsdx_sub = dsdx[:, self.active_dims_list[subspace_id]]

        return m_sub, np.sqrt(v_sub), dmdx_sub, dsdx_sub
