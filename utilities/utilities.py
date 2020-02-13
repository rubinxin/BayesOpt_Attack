# @author: Robin Ru (robin@robots.ox.ac.uk)

import numpy as np
from pyDOE import lhs
from scipy.optimize import fmin_l_bfgs_b

from utilities.upsampler import downsample_projection


def get_init_data(obj_func, n_init, bounds, method='lhs'):
    """
    Generate initial data for starting BO

    :param obj_func:
    :param n_init: number of initial data
    :param bounds: input space bounds
    :param method: random sample method
    :return x_init: initial input data
    :return y_init: initial output data
    """
    noise_var = 1.0e-10
    d = bounds.shape[0]

    if method == 'lhs':
        x_init = lhs(d, n_init) * (bounds[0, 1] - bounds[0, 0]) + bounds[0, 0]
    else:
        x_init = np.random.uniform(bounds[0, 0], bounds[0, 1], (n_init, d))
    f_init = obj_func(x_init)
    y_init = f_init + np.sqrt(noise_var) * np.random.randn(n_init, 1)
    return x_init, y_init


def subset_select(X_all, Y_all, select_metric='RAND'):
    """
    Select the subset of the observed data for sparse GP
    :param X_all: observed input data
    :param Y_all: observed output data
    :param select_metric: subset selection criterion
    :return X_ob: subset observed input data
    :return Y_ob: subset observed output data
    """

    N_ob = X_all.shape[0]

    if N_ob <= 500:
        X_ob = X_all
        Y_ob = Y_all
    else:
        # selecting subset if the number of observed data exceeds 500
        if N_ob > 500 and N_ob <= 1000:
            subset_size = 500
        else:
            subset_size = 1000

        print(f'use subset={subset_size} of observed data via {select_metric}')
        if 'SUBRAND' in select_metric:
            x_indices_random = np.random.permutation(range(N_ob))
            x_subset_indices = x_indices_random[:subset_size]
        elif 'SUBGREEDY' in select_metric:
            pseudo_prob_nexp = np.exp(-(Y_all - Y_all.min()))
            pseudo_prob = pseudo_prob_nexp / np.sum(pseudo_prob_nexp)
            x_subset_indices = np.random.choice(N_ob, subset_size, p=pseudo_prob.flatten(), replace=False)
        X_ob = X_all[x_subset_indices, :]
        Y_ob = Y_all[x_subset_indices, :]

    return X_ob, Y_ob


def subset_select_for_learning(X_all, Y_all, select_metric='ADDRAND'):
    """
    Select the subset of the observed data for sparse GP used only in the phase of learning dr or decomposition
    :param X_all: observed input data
    :param Y_all: observed output data
    :param select_metric: subset selection criterion
    :return X_ob: subset observed input data
    :return Y_ob: subset observed output data
    """
    N_ob = X_all.shape[0]
    subset_size = 200
    pseudo_prob_nexp = np.exp(-(Y_all - Y_all.min()))
    pseudo_prob = pseudo_prob_nexp / np.sum(pseudo_prob_nexp)
    x_subset_indices = np.random.choice(N_ob, subset_size, p=pseudo_prob.flatten(), replace=False)
    X_ob = X_all[x_subset_indices, :]
    Y_ob = Y_all[x_subset_indices, :]
    return X_ob, Y_ob


def optimise_acqu_func(acqu_func, bounds, X_ob, func_gradient=True, gridSize=10000, n_start=5):
    """
    Optimise acquisition function built on GP model

    :param acqu_func: acquisition function
    :param bounds: input space bounds
    :param X_ob: observed input data
    :param func_gradient: whether to use the acquisition function gradient in optimisation
    :param gridSize: random grid size
    :param n_start: the top n_start points in the random grid search from which we do gradient-based local optimisation
    :return np.array([opt_location]): global optimum input
    :return f_opt: global optimum
    """

    # Turn the acquisition function to be - acqu_func for minimisation
    target_func = lambda x: - acqu_func._compute_acq(x)

    # Define a new function combingin the acquisition function and its derivative
    def target_func_with_gradient(x):
        acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x)
        return -acqu_f, -dacqu_f

    # Define bounds for the local optimisers
    bounds_opt = list(bounds)

    # Create grid for random search
    d = bounds.shape[0]
    Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                           (gridSize, 1)) * np.random.rand(gridSize, d)
    Xgrid = np.vstack((Xgrid, X_ob))
    results = target_func(Xgrid)

    # Find the top n_start candidates from random grid search to perform local optimisation
    top_candidates_idx = results.flatten().argsort()[
                         :n_start]  # give the smallest n_start values in the ascending order
    random_starts = Xgrid[top_candidates_idx]
    f_min = results[top_candidates_idx[0]]
    opt_location = random_starts[0]

    # Print('done random grid search')
    # Perform multi-start gradient-based optimisation
    for random_start in random_starts:
        if func_gradient:
            x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds_opt,
                                            approx_grad=False, maxiter=5000)
        else:
            x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds_opt,
                                            approx_grad=True, maxiter=5000)
        if f_at_x < f_min:
            f_min = f_at_x
            opt_location = x

    f_opt = - f_min

    return np.array([opt_location]), f_opt


def optimise_acqu_func_mledr(acqu_func, bounds, X_ob, func_gradient=True, gridSize=10000, n_start=5):
    """
    Optimise acquisition function built on GP- model with learning dr

    :param acqu_func: acquisition function
    :param bounds: input space bounds
    :param X_ob: observed input data
    :param func_gradient: whether to use the acquisition function gradient in optimisation
    :param gridSize: random grid size
    :param n_start: the top n_start points in the random grid search from which we do gradient-based local optimisation
    :return np.array([opt_location]): global optimum input
    :return f_opt: global optimum
    """

    # Turn the acquisition function to be - acqu_func for minimisation
    target_func = lambda x: - acqu_func._compute_acq(x)

    # Define a new function combingin the acquisition function and its derivative
    def target_func_with_gradient(x):
        acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x)
        return -acqu_f, -dacqu_f

    # Define bounds for the local optimisers based on the optimal dr
    nchannel = acqu_func.model.nchannel
    d = acqu_func.model.opt_dr
    d_vector = int(acqu_func.model.opt_dr ** 2 * nchannel)
    bounds = np.vstack([[-1, 1]] * d_vector)

    # Project X_ob to optimal dr learnt
    h_d = int(X_ob.shape[1] / acqu_func.model.nchannel)
    X_ob_d_r = downsample_projection(acqu_func.model.dim_reduction, X_ob, int(d ** 2), h_d, nchannel=nchannel,
                                     align_corners=True)

    # Create grid for random search but split the grid into n_batches to avoid memory overflow
    good_results_list = []
    random_starts_candidates_list = []
    n_batch = 5
    gridSize_sub = int(gridSize / n_batch)
    for x_grid_idx in range(n_batch):
        Xgrid_sub = np.tile(bounds[:, 0], (gridSize_sub, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                                       (gridSize_sub, 1)) * np.random.rand(gridSize_sub,
                                                                                                           d_vector)
        if x_grid_idx == 0:
            Xgrid_sub = np.vstack((Xgrid_sub, X_ob_d_r))
        results = target_func(Xgrid_sub)
        top_candidates_sub = results.flatten().argsort()[:5]  # give the smallest n_start values in the ascending order
        random_starts_candidates = Xgrid_sub[top_candidates_sub]
        good_results = results[top_candidates_sub]
        random_starts_candidates_list.append(random_starts_candidates)
        good_results_list.append(good_results)

    # Find the top n_start candidates from random grid search to perform local optimisation
    results = np.vstack(good_results_list)
    X_random_starts = np.vstack(random_starts_candidates_list)
    top_candidates_idx = results.flatten().argsort()[
                         :n_start]  # give the smallest n_start values in the ascending order
    random_starts = X_random_starts[top_candidates_idx]
    f_min = results[top_candidates_idx[0]]
    opt_location = random_starts[0]

    # Perform multi-start gradient-based optimisation
    for random_start in random_starts:
        if func_gradient:
            x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds,
                                            approx_grad=False, maxiter=5000)
        else:
            x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds,
                                            approx_grad=True, maxiter=5000)
        if f_at_x < f_min:
            f_min = f_at_x
            opt_location = x

    f_opt = -f_min

    return np.array([opt_location]), f_opt


def optimise_acqu_func_additive(acqu_func, bounds, X_ob, func_gradient=True, gridSize=5000, n_start=1, nsubspace=12):
    """
    Optimise acquisition function built on ADDGP model

    :param acqu_func: acquisition function
    :param bounds: input space bounds
    :param X_ob: observed input data
    :param func_gradient: whether to use the acquisition function gradient in optimisation
    :param gridSize: random grid size
    :param n_start: the top n_start points in the random grid search from which we do gradient-based local optimisation
    :param nsubspace: number of subspaces in the decomposition
    :return np.array([opt_location]): global optimum input
    :return f_opt: global optimum
    """

    # Create grid for random search
    d = bounds.shape[0]
    Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                           (gridSize, 1)) * np.random.rand(gridSize, d)
    Xgrid = np.vstack((Xgrid, X_ob))
    f_opt_join = []

    # Get the learnt decomposition
    active_dims_list = acqu_func.model.active_dims_list
    opt_location_join_array = np.zeros(d)

    # Optimise the acquisition function in each subspace separately in sequence
    for i in range(nsubspace):
        print(f'start optimising subspace{i}')

        # Define the acquisition function for the subspace and turn it to be - acqu_func for minimisation
        def target_func(x_raw):
            x = np.atleast_2d(x_raw)
            N = x.shape[0]
            if x.shape[1] == d:
                x_aug = x.copy()
            else:
                x_aug = np.zeros([N, d])
                x_aug[:, active_dims_list[i]] = x
            return - acqu_func._compute_acq(x_aug, subspace_id=i)

        # Define a new function combingin the acquisition function and its derivative
        def target_func_with_gradient(x_raw):
            x = np.atleast_2d(x_raw)
            N = x.shape[0]
            if x.shape[1] == d:
                x_aug = x.copy()
            else:
                x_aug = np.zeros([N, d])
                x_aug[:, active_dims_list[i]] = x

            acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x_aug, subspace_id=i)
            return -acqu_f, -dacqu_f

        # Find the top n_start candidates from random grid search to perform local optimisation
        results = target_func(Xgrid)
        top_candidates_idx = results.flatten().argsort()[
                             :n_start]  # give the smallest n_start values in the ascending order
        random_starts = Xgrid[top_candidates_idx][:, active_dims_list[i]]
        f_min = results[top_candidates_idx[0]]
        opt_location = random_starts[0]

        # Define bounds for the local optimisers for the subspace
        bounds_opt_sub = list(bounds[active_dims_list[i], :])
        for random_start in random_starts:
            if func_gradient:
                x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds_opt_sub,
                                                approx_grad=False, maxiter=5000)
            else:
                x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds_opt_sub,
                                                approx_grad=True, maxiter=5000)
            if f_at_x < f_min:
                f_min = f_at_x
                opt_location = x

        f_opt = -f_min
        opt_location_join_array[active_dims_list[i]] = opt_location
        f_opt_join.append(f_opt)

    f_opt_join_sum = np.sum(f_opt_join)

    return np.atleast_2d(opt_location_join_array), f_opt_join_sum


def optimise_acqu_func_for_NN(acqu_func, bounds, X_ob, func_gradient=False, gridSize=20000, num_chunks=10):
    """
    Optimise acquisition function built on BNN surrogate

    :param acqu_func: acquisition function
    :param bounds: input space bounds
    :param X_ob: observed input data
    :param func_gradient: whether to use the acquisition function gradient in optimisation
    :param gridSize: random grid size
    :param num_chunks: divide the random grid into a number of chunks to avoid memory overflow
    :return np.array([opt_location]): global optimum input
    :return f_opt: global optimum
    """

    # Turn the acquisition function to be - acqu_func for minimisation
    target_func = lambda x: - acqu_func._compute_acq(x)

    # Create grid for random search but split the grid into num_chunks to avoid memory overflow
    d = bounds.shape[0]
    Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                           (gridSize, 1)) * np.random.rand(gridSize, d)
    X_chunks = np.split(Xgrid, num_chunks)
    x_ob_chunk = X_ob[-int(gridSize / num_chunks):, :]
    X_chunks.append(x_ob_chunk)
    Xgrid = np.vstack((Xgrid, x_ob_chunk))
    results_list = [target_func(x_chunk) for x_chunk in X_chunks]
    results = np.vstack(results_list)

    # Find the top candidate from random grid search
    top_candidates_idx = results.flatten().argsort()[:1]  # give the smallest n_start values in the ascending order
    random_starts = Xgrid[top_candidates_idx]
    print('done selecting rs')
    f_min = results[top_candidates_idx[0]]
    opt_location = random_starts[0]
    f_opt = -f_min

    return np.array([opt_location]), f_opt
