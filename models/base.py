from abc import abstractmethod

class BaseModel:

    @abstractmethod
    def _create_model(self, X, Y):
        raise NotImplementedError('')

    @abstractmethod
    def _update_model(self, X_all, Y_all, itr=0):
        """
        Updates the model with new observations.
        """
        return

    @abstractmethod
    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X.
        """
        return

    @abstractmethod
    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        return
