# @author: Robin Ru (robin@robots.ox.ac.uk)

import GPy
import numpy as np

from utilities.upsampler import downsample_projection
from utilities.utilities import subset_select, subset_select_for_learning
from .base import BaseModel


class GPModelLDR(BaseModel):

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs',
                 max_iters=1000, optimize_restarts=5, sparse=None, num_inducing=10,
                 verbose=False, ARD=False, seed=42, normalize_Y=True, update_freq=5,
                 high_dim=int(32 * 32), nchannel=3, dim_reduction='BILI'):
        """
        GP model which also learns the reduced dimension dr

        :param kernel: the GP kernel option; use default if set to None
        :param noise_var: observation noise variance
        :param exact_feval: whether to learn observation noise (exact_feval=False) or not (exact_feval=True)
        :param optimizer: to optimise marginal likelihood w.r.t GP hyperparameters
        :param max_iters: max iterations for optimising marginal likelihood
        :param optimize_restarts: number of restart for optimising marginal likelihood
        :param sparse:
        :param num_inducing: number of inducing points if a sparse GP is used
        :param verbose: enable print-statements
        :param ARD: whether ARD is used in the kernel (default, False).
        :param seed: set ranfom seed
        :param normalize_Y: normalise output data
        :param update_freq: frequency of relearning GP hyperparameters
        :param high_dim: image dimension (e.g. 32x32 for CIFAR10) or high-dimensional search space for imagenet (96x96)
        :param nchannel: number of image channels
        :param dim_reduction: dimension reduction method used in upsampling
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
        if high_dim > int(40 * 40):
            self.dr_list = list(range(6, 61, 6))
        else:
            self.dr_list = list(range(6, 21, 2))
        self.high_dim = high_dim
        self.nchannel = nchannel
        self.dim_reduction = dim_reduction
        np.random.rand(self.seed)

    def _create_model(self, X, Y):
        """
        :param X: observed input data
        :param Y: observed output data
        :return model: GP model
        """
        # define GP kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD)
        else:
            kern = self.kernel
            self.kernel = None

        # define GP model
        noise_var = Y.var() * 0.01 if self.noise_var is None else self.noise_var
        if not self.sparse_surrogate:
            model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        else:
            model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing)

        if self.exact_feval:
            # restrict noise variance if exact evaluations of the objective
            model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # bound the noise variance if not
            model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False)

        try:
            model.optimize(optimizer=self.optimizer, max_iters=self.max_iters, messages=False,
                           ipython_notebook=False)
        except:
            model = model

        return model

    def _update_model(self, X_all, Y_all_raw, itr=0):
        """
        :param X_all: observed input data
        :param Y_all_raw: observed output raw data
        :param itr: BO iteration counter
        """

        # normalise the observed output raw data
        if self.normalize_Y:
            Y_all = (Y_all_raw - Y_all_raw.mean()) / (Y_all_raw.std())
            self.Y_mean = Y_all_raw.mean()
            self.Y_std = Y_all_raw.std()
        else:
            Y_all = Y_all_raw

        # if use sparse GP option, select only a subset of observed data for training surrogate (one-type of sparse GP)
        if self.sparse.startswith('SUB'):
            X_all, Y_all = subset_select(X_all, Y_all, select_metric=self.sparse)

        if itr % int(8 * self.update_interval) == 0:
            # learn the optimal dr at an interval of (8 * interval of relearning hyperparameter)
            if 'ADD' in self.sparse:
                print('learn the optimal dr with subset observed data')
                X_ob, Y_ob = subset_select_for_learning(X_all, Y_all, select_metric=self.sparse)
            else:
                X_ob, Y_ob = X_all, Y_all

            ll_list = []
            model_list = []
            for dr in self.dr_list:
                X_all_d_r = downsample_projection(self.dim_reduction, X_ob, int(dr ** 2), self.high_dim,
                                                  nchannel=self.nchannel, align_corners=True)
                model = self._create_model(X_all_d_r, Y_ob)
                ll_dr = model.log_likelihood()
                print(f'dr={dr}, ll={ll_dr}')
                ll_list.append(ll_dr)
                model_list.append(model)

            # select the dr with the maximum marginal likelihood
            mle_idx = np.argmax(ll_list)
            self.opt_dr = int(self.dr_list[mle_idx])
            self.model = model_list[mle_idx]
            print(f'opt_dr={self.opt_dr}')

        else:
            # downsample the observed input data to the previous optimal dr
            X_all_d_r = downsample_projection(self.dim_reduction, X_all, int(self.opt_dr ** 2), self.high_dim,
                                              nchannel=self.nchannel,
                                              align_corners=True)
            self.model.set_XY(X_all_d_r, Y_all)

        if itr % self.update_interval == 0:
            # relearn the GP hyperparameters at a certain iteration interval
            self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer,
                                         max_iters=self.max_iters, verbose=self.verbose)

    def predict(self, X):
        """
        :param X: test location
        :return m: predictive posterior mean
        :return s: predictive posterior standard deviation
        """

        if X.ndim == 1: X = X[None, :]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)

        return m, np.sqrt(v)

    def predict_withGradients(self, X):
        """
        :param X: test location
        :return m: predictive posterior mean
        :return s: predictive posterior standard deviation
        :return dmdx: derivative of predictive posterior mean
        :return dsdx: derivative of predictive posterior standard deviation
        """

        if X.ndim == 1: X = X[None, :]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:, :, 0]
        dsdx = dvdx / (2 * np.sqrt(v))

        return m, np.sqrt(v), dmdx, dsdx
