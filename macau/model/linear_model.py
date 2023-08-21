import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import minimize
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, PassiveAggressiveClassifier, ARDRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
            
class DeltaLogisticRegression:
    """
    Classifier with confidence interval estimation delta method.
    """

    def __init__(self):
        """
        Initialize the LogisticRegressionWithDeltaMethod.
        """
        self._model = LogisticRegression()
        self._cov = None


    def _fit_delta(self, X, Y):
        """
        Fit the classifier using the delta method.

        Parameters:
        - X: Input data.
        - Y: Target labels.

        Returns:
        - self
        """
        if len(np.unique(Y)) == 1:
            unique = np.unique(Y)
            dummy = DummyClassifierImpl(strategy='constant', constant=unique[0])
            dummy.fit(np.zeros((max(2, len(unique)), X.shape[1])), np.arange(max(2, len(unique)))) 
            self._model = dummy
        else:
            self._model.fit(X, Y)
            if not hasattr(self._model, 'coef_'):
                self._model.coef_ = np.array([list(np.zeros(X.shape[1]))])
        self._cov = self._covariance(X, self._model)
        return self
    
    def _covariance(self, X, model):
        """
        Compute the covariance matrix using the model's predicted probabilities.

        Parameters:
        - X: Input data.
        - model: Classification model.

        Returns:
        - Covariance matrix.
        """
        V = np.product(model.predict_proba(X), axis=1)    
        return np.linalg.pinv(np.dot(X.T * V, X))
    
    def _to_ndarray(self, X):
        """
        Convert input to numpy ndarray if necessary.

        Parameters:
        - X: Input data.

        Returns:
        - Numpy ndarray.
        """
        if not isinstance(X, np.ndarray):
            return np.array(X)
        return X

    def fit(self, X, Y, verbose=0):
        """
        Fit the classifier to the training data.

        Parameters:
        - X: Input data.
        - Y: Target labels.
        - verbose: Verbosity level (default: 0).

        Returns:
        - self
        """
        X = self._to_ndarray(X)
        Y = self._to_ndarray(Y)

        self._fit_delta(X, Y)
        return self
    
    def _predict(self, X):
        """
        Perform prediction using the delta method.

        Parameters:
        - X: Input data.

        Returns:
        - Tuple of class probabilities and their standard deviations.
        """
        
        proba = self._model.predict_proba(X)
        gradient = np.multiply(np.product(proba, axis=1)[:, np.newaxis], X)
        sigma = np.sqrt(np.einsum('ij,jk,ik->i', gradient, self._cov, gradient))
        return proba, sigma
        
    def predict_proba(self, X, return_std=False):
        """
        Compute class probabilities for X.

        Parameters:
        - X: Input data.
        - return_std: Whether to return standard deviations (default: False).

        Returns:
        - Class probabilities.
        """
        proba, std = self._predict(X)
        if return_std:
            return proba, std
        else:
            return proba
        
    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters:
        - X: Input data.

        Returns:
        - Predicted class labels.
        """
        return np.argmax(self.predict_proba(X), axis=1)
            
class ARDUncertaintyRegressor:
    def __init__(self, n_estimators=10):
        self._n_estimators = n_estimators
        self._models = [ARDRegression() for _ in range(n_estimators)]

    def fit(self, X, Y):
        for model in self._models:
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            model.fit(X[indices], Y[indices])
        
        return self

    def predict(self, X, return_std = False):
        y_preds = np.zeros((self._n_estimators, X.shape[0]))
        y_stds = np.zeros((self._n_estimators, X.shape[0]))

        for i, model in enumerate(self._models):
            preds, stds = model.predict(X, return_std = True)
            y_preds[i] = preds
            y_stds[i] = stds
        aleatoric_std = np.mean(y_stds, axis=0)  # Aleatoric uncertainty
        epistemic_std = np.std(y_preds, axis=0)  # Epistemic uncertainty (approximation)
        y_pred = np.mean(y_preds, axis=0)  # Mean prediction
        if return_std:
            return y_pred, aleatoric_std + epistemic_std
        else:
            return y_pred

class DummyClassifierImpl(DummyClassifier):
    """
    Extension of scikit-learn's DummyClassifier that allows setting coefficients explicitly.
    """

    def __init__(self, **args):
        super().__init__(**args)
        self._coef = None

    @property
    def coef_(self):
        """Get the coefficients of the classifier."""
        return self._coef

    @coef_.setter
    def coef_(self, coef):
        """Set the coefficients of the classifier."""
        self._coef = coef

    def fit(self, X, Y, **args):
        """Fit the classifier to the training data."""
        super().fit(X, Y, **args)
        self._coef = np.array([list(np.zeros(X.shape[1]))])
        return self
        
class DummyRegressorCI(DummyRegressor):
    """
    Extension of scikit-learn's DummyRegressor that allows setting a standard deviation for predictions.
    """

    def __init__(self, **args):
        super().__init__(**args)
        self._std = None

    def fit(self, X, y, std=None, sample_weight=None):
        """
        Fit the regressor to the training data.
        
        Parameters:
        - X: Input data.
        - y: Target values.
        - std: Standard deviation for predictions.
        - sample_weight: Sample weights.
        """
        DummyRegressor.fit(self, X, y, sample_weight)
        if std is not None:
            self._std = std
        else:
            self._std = np.std(y)            
        return self

    def predict(self, X, return_std=False):
        """
        Predict target values for X.
        
        Parameters:
        - X: Input data.
        - return_std: If True, return predictions along with the standard deviation.
        
        Returns:
        - Predicted target values.
        - Standard deviation (if return_std=True).
        """
        if return_std and self._std is not None:
            return DummyRegressor.predict(self, X), self._std * np.ones(X.shape[0])
        else:
            return DummyRegressor.predict(self, X)

class DummyClassifierCI:
    """
    Dummy classifier that allows specifying a constant prediction and standard deviation.
    """

    def __init__(self, constant=0.5, std=0.0):
        """
        Initialize the DummyClassifierCI.

        Parameters:
        - constant: Constant value for predictions.
        - std: Standard deviation for predictions.
        """
        self._constant = constant
        self._std = std
        self.coef_ = None

    def fit(self, X, y, **args):
        """
        Fit the classifier to the training data.

        Parameters:
        - X: Input data.
        - y: Target values.

        Returns:
        - self: Returns an instance of the fitted classifier.
        """
        self.coef_ = np.array([list(np.zeros(X.shape[1]))])
        return self
    
    def predict_proba(self, X, return_std=False):
        """
        Compute class probabilities for X.

        Parameters:
        - X: Input data.
        - return_std: If True, return probabilities along with the standard deviation.

        Returns:
        - Class probabilities.
        - Standard deviation (if return_std=True).
        """
        if return_std:
            return np.vstack([(1 - self._constant) * np.ones(X.shape[0]), 
                              self._constant * np.ones(X.shape[0])]).T, self._std * np.ones(X.shape[0])
        else:
            return np.vstack([(1 - self._constant) * np.ones(X.shape[0]), 
                              self._constant * np.ones(X.shape[0])]).T
    
    def predict(self, X, return_std=False):
        """
        Predict target values for X.

        Parameters:
        - X: Input data.
        - return_std: If True, return predictions along with the standard deviation.

        Returns:
        - Predicted target values.
        - Standard deviation (if return_std=True).
        """
        if return_std and self._std is not None:
            proba, std = self.predict_proba(X, return_std)
            return proba[:, 1] > 0.5, std
        else:
            return self.predict_proba(X)[:, 1] > 0.5

class PAC(PassiveAggressiveClassifier):
    """
    Passive Aggressive Classifier with modified predict_proba method.
    """

    def __init__(self, **args):
        """
        Initialize the PAC classifier.

        Parameters:
        - args: Additional arguments to be passed to the superclass constructor.
        """
        super().__init__(**args)
    
    def predict_proba(self, X):
        """
        Compute class probabilities for X.

        Parameters:
        - X: Input data.

        Returns:
        - Class probabilities.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
        
class ClassifierCI:
    """
    Classifier with confidence interval estimation using bootstrap or delta method.
    """

    def __init__(self, bootstrap=5, model=None):
        """
        Initialize the ClassifierCI.

        Parameters:
        - bootstrap: Number of bootstrap iterations (default: 5).
        - model: Base classification model (default: LogisticRegression()).
        """
        self._bootstrap = bootstrap
        self._model = model or LogisticRegression()
        self._models = []
        self._cov = None

    @property
    def model(self):
        """
        Get the current base classification model.

        Returns:
        - Base classification model.
        """
        return self._model

    @model.setter
    def model(self, model):
        """
        Set the base classification model.

        Parameters:
        - model: Base classification model.
        """
        self._model = model

    @property
    def coef_(self):
        """
        Get the coefficients of the classifier.

        Returns:
        - Coefficients of the classifier.
        """
        if self.use_bootstrap:
            return np.mean([model.coef_ for model in self._models], axis=0).ravel()
        else:
            return self._model.coef_.ravel()

    @property
    def use_bootstrap(self):
        """
        Check if the classifier uses bootstrap for confidence interval estimation.

        Returns:
        - Boolean indicating if bootstrap is used.
        """
        return self._bootstrap > 1

    
    def _fit_bootstrap(self, X, Y, verbose=0, **args):
        """
        Fit the classifier using bootstrap.

        Parameters:
        - X: Input data.
        - Y: Target labels.
        - verbose: Verbosity level (default: 0).
        - args: Additional arguments.

        Returns:
        - self
        """
        self._models = []                
        for i in tqdm(range(self._bootstrap), disable=verbose == 0, desc='Fitting bootstraps'):
            X_resampled, Y_resampled = resample(X, Y)
            
            if len(np.unique(Y_resampled)) > 1:
                self._models.append(clone(self._model).fit(X_resampled, Y_resampled))
                if not hasattr(self._models[-1], 'coef_'):
                    self._models[-1].coef_ = np.array([list(np.zeros(X_resampled.shape[1]))])
            else:
                unique = Y_resampled[0]
                dummy = DummyClassifierImpl(strategy='constant', constant=unique)
                dummy.fit(np.zeros((2, X_resampled.shape[1])), np.arange(2))
                self._models.append(dummy)
        return self

    def _fit_delta(self, X, Y):
        """
        Fit the classifier using the delta method.

        Parameters:
        - X: Input data.
        - Y: Target labels.

        Returns:
        - self
        """
        if len(np.unique(Y)) == 1:
            unique = np.unique(Y)
            dummy = DummyClassifierImpl(strategy='constant', constant=unique[0])
            dummy.fit(np.zeros((max(2, len(unique)), X.shape[1])), np.arange(max(2, len(unique)))) 
            self._model = dummy
        else:
            self._model.fit(X, Y)
            if not hasattr(self._model, 'coef_'):
                self._model.coef_ = np.array([list(np.zeros(X.shape[1]))])
        self._cov = self._covariance(X, self._model)
        return self
    
    
    def _covariance(self, X, model):
        """
        Compute the covariance matrix using the model's predicted probabilities.

        Parameters:
        - X: Input data.
        - model: Classification model.

        Returns:
        - Covariance matrix.
        """
        V = np.product(model.predict_proba(X), axis=1)    
        return np.linalg.pinv(np.dot(X.T * V, X))
    
    def _to_ndarray(self, X):
        """
        Convert input to numpy ndarray if necessary.

        Parameters:
        - X: Input data.

        Returns:
        - Numpy ndarray.
        """
        if not isinstance(X, np.ndarray):
            return np.array(X)
        return X

    def fit(self, X, Y, verbose=0):
        """
        Fit the classifier to the training data.

        Parameters:
        - X: Input data.
        - Y: Target labels.
        - verbose: Verbosity level (default: 0).

        Returns:
        - self
        """
        X = self._to_ndarray(X)
        Y = self._to_ndarray(Y)

        if len(np.unique(Y)) > 2:
            self._bootstrap = 5

        if self.use_bootstrap:
            self._fit_bootstrap(X, Y, verbose=verbose)
        else:
            self._fit_delta(X, Y)
        return self
    
    
    def _predict_bootstrap(self, X):
        """
        Perform prediction using bootstrap.

        Parameters:
        - X: Input data.

        Returns:
        - Tuple of class probabilities and their standard deviations.
        """
        probas = np.array([model.predict_proba(X) for model in self._models])
        if probas.shape[2] > 2:
            return np.mean(probas, axis=0), np.std(probas, axis=0)
        else:
            return np.mean(probas, axis=0), np.mean(np.std(probas, axis=0), axis=1)
        
    def _predict_delta(self, X):
        """
        Perform prediction using the delta method.

        Parameters:
        - X: Input data.

        Returns:
        - Tuple of class probabilities and their standard deviations.
        """
        
        proba = self._model.predict_proba(X)
        gradient = np.multiply(np.product(proba, axis=1)[:, np.newaxis], X)
        sigma = np.sqrt(np.einsum('ij,jk,ik->i', gradient, self._cov, gradient))
        return proba, sigma
        
    def _predict(self, X):
        """
        Perform prediction using the appropriate method (bootstrap or delta).

        Parameters:
        - X: Input data.

        Returns:
        - Tuple of class probabilities and their standard deviations.
        """
        if self.use_bootstrap:
            return self._predict_bootstrap(X)
        else:
            return self._predict_delta(X)
        
    def predict_proba(self, X, return_std=False):
        """
        Compute class probabilities for X.

        Parameters:
        - X: Input data.
        - return_std: Whether to return standard deviations (default: False).

        Returns:
        - Class probabilities.
        """
        proba, std = self._predict(X)
        if return_std:
            return proba, std
        else:
            return proba
        
    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters:
        - X: Input data.

        Returns:
        - Predicted class labels.
        """
        return np.argmax(self.predict_proba(X), axis=1)

class UncertaintyAwareRegressor:
    def __init__(self, model=None, sigma_model=None, fit_primary=True):
        """
        A regression model that estimates uncertainty along with predictions.

        Parameters:
        - model: The primary regression model (default: LinearRegression()).
        - sigma_model: The model for estimating uncertainty (default: LinearRegression()).
        - fit_primary: Whether to fit the primary model during training (default: True).
        """
        if model is None:
            self._fit_primary = True
            self._model = LinearRegression()
        else:
            self._model = model
        if sigma_model is None:
            self._sigma_model = LinearRegression()
        else:
            self._sigma_model = sigma_model
        self._fit_primary = fit_primary
        
    @property
    def coef_(self):
        """
        Coefficients of the primary regression model.

        Returns:
        - Coefficients.
        """
        return self._model.coef_
    
    def _loss(self, Y, Y_pred, pred_sigma):
        """
        Loss function for uncertainty estimation.

        Parameters:
        - Y: True labels.
        - Y_pred: Predicted labels.
        - pred_sigma: Predicted uncertainties.

        Returns:
        - Loss value.
        """
        return np.mean(2 * pred_sigma + np.square((Y - Y_pred) / np.exp(pred_sigma)))
    
    def _min_fun(self, coefs, X, Y, Y_pred):            
        """
        Objective function to minimize during fitting.

        Parameters:
        - coefs: Model coefficients.
        - X: Input data.
        - Y: True labels.
        - Y_pred: Predicted labels.

        Returns:
        - Loss value.
        """
        self._sigma_model.coef_ = coefs[:-1]
        self._sigma_model.intercept_ = coefs[-1]                    
        pred_sigma = self._sigma_model.predict(X) 
        return self._loss(Y, Y_pred, pred_sigma)
    
    def fit(self, X, Y):
        """
        Fit the uncertainty-aware regressor to the training data.

        Parameters:
        - X: Input data.
        - Y: True labels.

        Returns:
        - self
        """
        if self._fit_primary:
            self._model = self._model.fit(X, Y)
        coefs = np.zeros(X.shape[1] + 1)
        coefs = minimize(self._min_fun, coefs, args=(X, Y, self._model.predict(X))).x
        self._sigma_model.coef_ = coefs[:-1]
        self._sigma_model.intercept_ = coefs[-1]
        return self
    
    def predict(self, X, return_std=False):
        """
        Perform regression on samples in X.

        Parameters:
        - X: Input data.
        - return_std: Whether to return standard deviations (default: False).

        Returns:
        - Predicted labels.
        """
        if return_std:
            return self._model.predict(X), np.exp(self._sigma_model.predict(X))
        else:
            return self._model.predict(X)


class StandardErrorRegressor:
    def __init__(self, model=None, fit_primary=True):
        """
        A regression model that estimates standard error along with predictions.

        Parameters:
        - model: The primary regression model (default: LinearRegression()).
        - fit_primary: Whether to fit the primary model during training (default: True).
        """
        if model is None:
            self._fit_primary = True
            self._model = LinearRegression()
        else:
            self._model = model
        self._fit_primary = fit_primary
        self._error = 0
        
    @property
    def coef_(self):
        """
        Coefficients of the primary regression model.

        Returns:
        - Coefficients.
        """
        return self._model.coef_
    
    def fit(self, X, Y):
        """
        Fit the standard error regressor to the training data.

        Parameters:
        - X: Input data.
        - Y: True labels.

        Returns:
        - self
        """
        if self._fit_primary:
            self._model = self._model.fit(X, Y)
        self._error = mean_squared_error(Y, self._model.predict(X))**0.5
        return self
    
    def predict(self, X, return_std=False):
        """
        Perform regression on samples in X.

        Parameters:
        - X: Input data.
        - return_std: Whether to return standard errors (default: False).

        Returns:
        - Predicted labels.
        """
        if return_std:
            return self._model.predict(X), np.array([self._error] * X.shape[0])
        else:
            return self._model.predict(X)
