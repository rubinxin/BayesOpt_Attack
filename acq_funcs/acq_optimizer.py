"""
Created on Fri Nov 10 13:45:16 2017
@author: robin
"""
import numpy as np

from utilities.utilities import optimise_acqu_func, optimise_acqu_func_for_NN, optimise_acqu_func_additive, \
    optimise_acqu_func_mledr


class Acq_Optimizer(object):

    def __init__(self, model, acqu_func, bounds, batch_method='CL', batch_size=1, model_name='GP', nsubspace=1):
        """
        Optimise the acquisition functions to recommend the next (batch) locations for evaluation

        :param model: BO surrogate model function
        :param acqu_func: BO acquisition function
        :param bounds: input space bounds
        :param batch_method: the method for selecting a batch of new locations to be evaluated next
        :param batch_size: the number of new query locations in the batch (=1 for sequential BO and > 1 for parallel BO)
        :param model_name: the name of the BO surrogate model
        :param nsubspace: number of subspaces needs to be specified for ADDGP-BO but equals 1 for other BO attacks
        """
        self.model = model
        self.acqu_func = acqu_func
        self.batch_size = batch_size
        self.batch_method = batch_method
        self.bounds = bounds
        self.model_name = model_name
        self.nsubspace = nsubspace

    def get_next(self, X, Y):
        """
        :param X: observed input data
        :param Y: observed output data
        :return X_batch: batch of inputs recommended by BO to be evaluated next
        :return batch_acq_value: acqusitioin function values of the batch of inputs recommended
        """

        # BO batch method: Constant Liar
        if self.batch_method.upper() == 'CL':

            if self.model_name == 'GP':
                X_batch, batch_acq_value = optimise_acqu_func(acqu_func=self.acqu_func, bounds=self.bounds, X_ob=X)
            elif self.model_name == 'GPLDR':
                X_batch, batch_acq_value = optimise_acqu_func_mledr(acqu_func=self.acqu_func, bounds=self.bounds,
                                                                    X_ob=X)
            elif self.model_name.startswith('ADDGP'):
                X_batch, batch_acq_value = optimise_acqu_func_additive(acqu_func=self.acqu_func, bounds=self.bounds,
                                                                       X_ob=X, nsubspace=self.nsubspace)
            else:
                X_batch, batch_acq_value = optimise_acqu_func_for_NN(acqu_func=self.acqu_func, bounds=self.bounds,
                                                                     X_ob=X)

            new_batch_point = X_batch
            temporal_X = np.copy(X)
            temporal_Y = np.copy(Y)

            # Assume the functional value at last query location is equal to a constant
            L = np.min(temporal_Y)
            # Get the remaining points in the batch
            k = 1

            while k < self.batch_size:
                # Augment the observed data with previous query location and the constant liar L
                temporal_X = np.vstack((temporal_X, new_batch_point))
                temporal_Y = np.vstack((temporal_Y, L))

                # Update the surrogate model (no update on hyperparameter) and acq_func with the augmented observation data
                self.model._update_model(temporal_X, temporal_Y)

                if self.model_name == 'GP':
                    new_batch_point, next_batch_acq_value = optimise_acqu_func(acqu_func=self.acqu_func,
                                                                               bounds=self.bounds, X_ob=X)

                elif self.model_name == 'GPLDR':
                    new_batch_point, next_batch_acq_value = optimise_acqu_func_mledr(acqu_func=self.acqu_func,
                                                                                     bounds=self.bounds, X_ob=X)

                elif self.model_name.startswith('ADDGP'):
                    new_batch_point, next_batch_acq_value = optimise_acqu_func_additive(acqu_func=self.acqu_func,
                                                                                        bounds=self.bounds, X_ob=X,
                                                                                        nsubspace=self.nsubspace)
                else:
                    new_batch_point, next_batch_acq_value = optimise_acqu_func_for_NN(acqu_func=self.acqu_func,
                                                                                      bounds=self.bounds, X_ob=X)

                X_batch = np.vstack((X_batch, new_batch_point))
                batch_acq_value = np.vstack((batch_acq_value, next_batch_acq_value))
                k += 1

        # BO batch method: Kriging Believer
        elif self.batch_method.upper() == 'KB':

            if self.model_name == 'GP':
                X_batch, batch_acq_value = optimise_acqu_func(acqu_func=self.acqu_func, bounds=self.bounds, X_ob=X)
            elif self.model_name.startswith('ADDGP'):
                X_batch, batch_acq_value = optimise_acqu_func_additive(acqu_func=self.acqu_func, bounds=self.bounds,
                                                                       X_ob=X, nsubspace=self.nsubspace)
            elif self.model_name == 'GPLDR':
                X_batch, batch_acq_value = optimise_acqu_func_mledr(acqu_func=self.acqu_func, bounds=self.bounds,
                                                                    X_ob=X)
            else:
                X_batch, batch_acq_value = optimise_acqu_func_for_NN(acqu_func=self.acqu_func, bounds=self.bounds,
                                                                     X_ob=X, func_gradient=False)

            new_batch_point = X_batch
            temporal_X = np.copy(X)
            temporal_Y = np.copy(Y)

            # Get the remaining points in the batch if batch_size > 1
            k = 1
            while self.batch_size > 1:

                # Believe the predictor: the functional value at last query location is equal to its predicitve mean
                mu_new_batch_point, _ = self.model.predict(new_batch_point)

                # Augment the observed data with previous query location and the preditive mean at that location
                temporal_X = np.vstack((temporal_X, new_batch_point))
                temporal_Y = np.vstack((temporal_Y, mu_new_batch_point))

                # Update the surrogate model (no update on hyperparameter) and acq_func with the augmented observation data
                self.model._update_model(temporal_X, temporal_Y)

                if self.model_name == 'GP':
                    new_batch_point, next_batch_acq_value = optimise_acqu_func(acqu_func=self.acqu_func,
                                                                               bounds=self.bounds, X_ob=X)
                elif self.model_name.startswith('ADDGP'):
                    new_batch_point, next_batch_acq_value = optimise_acqu_func_additive(acqu_func=self.acqu_func,
                                                                                        bounds=self.bounds, X_ob=X,
                                                                                        nsubspace=self.nsubspace)
                elif self.model_name == 'GPLDR':
                    new_batch_point, next_batch_acq_value = optimise_acqu_func_mledr(acqu_func=self.acqu_func,
                                                                                     bounds=self.bounds, X_ob=X)
                else:
                    new_batch_point, next_batch_acq_value = optimise_acqu_func_for_NN(acqu_func=self.acqu_func,
                                                                                      bounds=self.bounds, X_ob=X)

                X_batch = np.vstack((X_batch, new_batch_point))
                batch_acq_value = np.append(batch_acq_value, next_batch_acq_value)
                k += 1

        return X_batch, batch_acq_value
