import math
import numpy as np

class SGDL2RegularizedLinearModel(object):
    """
    An L2-regularized linear model that uses SGD to minimize the in-sample error function.
    """
    
    def __init__(self, n_epochs, learning_rate, batch_size, regularization_strength=0.0):
        """
        Initialize the linear model.
        
        n_epochs -- int.
        learning_rate -- float or function. If it is a function, it is called with one argument 't',
                         where 't' is the iteration counter.
        batch_size -- int. If None, the batch_size is the number of examples in the training set
                      (regular gradient descent).
        regularization_strength -- float. 0.0 (no regularization) by default.
        """
        self._w = None
        self._n_epochs = n_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._regularization_strength = regularization_strength
            
    def gradient(self, X, y):
        """
        Return the gradient of the in-sample error function as a numpy array of shape
        [n_features, 1].
        
        X -- numpy array of shape [n_samples, n_features].
        y -- numpy array of shape [n_samples].
        """
        raise NotImplementedError
        
    def predict(self, X):
        """
        Return predicted values as a numpy array of shape [n_samples].
        
        X -- numpy array of shape [n_samples, n_features].
        """
        raise NotImplementedError        
            
    def fit(self, X, y):
        """
        Fit the model with training data.
        
        X -- numpy array of shape [n_samples, n_features].
        y -- numpy array of shape [n_samples].
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # x_0 is always 1
        self._w = np.random.randn(X.shape[1], 1)
        batch_size = self._batch_size if self._batch_size is not None else X.shape[0]
        for i in range(self._n_epochs):
            for j in range(X.shape[0] / batch_size):
                learning_rate = self._learning_rate if isinstance(self._learning_rate, float) \
                                else self._learning_rate(i * (X.shape[0] / batch_size) + j)
                sample = np.random.choice(X.shape[0], batch_size, replace=False)
                self._w -= learning_rate * self.gradient(X[sample,:], y[sample])

class SGDLinearRegression(SGDL2RegularizedLinearModel):
    """
    A linear regression that uses SGD to minimize the in-sample error function.
    """

    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape((np.dot(np.transpose(self._w), xi) - yi) * xi, (X.shape[1], 1))
        return gradient * (2.0 / X.shape[0])

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # x_0 is always 1
        return np.dot(np.transpose(self._w), np.transpose(X)).flatten()

class SGDRidgeRegression(SGDL2RegularizedLinearModel):
    """
    A Ridge regression that uses SGD to minimize the in-sample error function.
    """

    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape((np.dot(np.transpose(self._w), xi) - yi) * xi, (X.shape[1], 1))
        gradient *= (2.0 / X.shape[0])
        return gradient + 2.0 * self._regularization_strength * self._w
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # x_0 is always 1
        return np.dot(np.transpose(self._w), np.transpose(X)).flatten()

class SGDLogisticRegression(SGDL2RegularizedLinearModel):
    """
    A logistic regression that uses SGD to minimize the in-sample error function.
    """

    def theta(self, s):
        return (math.e ** s) / (1 + math.e ** s)
    
    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape(self.theta(-yi * np.dot(np.transpose(self._w), xi)) * yi * xi,
                                   (X.shape[1], 1))
        return gradient * (-1.0 / X.shape[0])
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # x_0 is always 1
        return np.vectorize(lambda x: self.theta(x))(
                np.dot(np.transpose(self._w), np.transpose(X)).flatten()
        )

class SGDL2RegularizedLogisticRegression(SGDL2RegularizedLinearModel):
    """
    An L2 regularized logistic regression that uses SGD to minimize the in-sample error function.
    """

    def theta(self, s):
        return (math.e ** s) / (1 + math.e ** s)
    
    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape(self.theta(-yi * np.dot(np.transpose(self._w), xi)) * yi * xi,
                                   (X.shape[1], 1))
        gradient *= (-1.0 / X.shape[0])
        return gradient + 2.0 * self._regularization_strength * self._w
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # x_0 is always 1
        return np.vectorize(lambda x: self.theta(x))(
                np.dot(np.transpose(self._w), np.transpose(X)).flatten()
        )
