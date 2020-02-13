# @author: Robin Ru (robin@robots.ox.ac.uk)
# modified based on GP model in GpyOpt

import GPy
import numpy as np

from utilities.utilities import subset_select
from .base import BaseModel


class GPModel(BaseModel):

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs',
                 max_iters=1000, optimize_restarts=5, sparse=None, num_inducing=10,
                 verbose=False, ARD=False, seed=42, normalize_Y=True, update_freq=5):
        """
        GP model (modified based on the GP model in GPyOpt).

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
        np.random.rand(self.seed)

    def _create_model(self, X, Y):
        """
        :param X: observed input data
        :param Y: observed output data
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
            self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        else:
            self.model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing)

        if self.exact_feval:
            # restrict noise variance if exact evaluations of the objective
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # bound the noise variance if not
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False)

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

        # initialise model or update the observed data for the model
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

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
