import numpy as np
from scipy.linalg import norm
import copy
from sklearn.covariance import LedoitWolf, MinCovDet, EllipticEnvelope, OAS
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from macau.model.linear_model import DummyClassifierImpl

class RobustPCA(PCA):
    """
    Robust Principal Component Analysis (PCA) implementation.

    This class extends the scikit-learn PCA class and adds robustness to the
    estimation of principal components by using a robust covariance estimator.

    Parameters:
    - n_components: Number of components to keep (default: None).
    - **kwargs: Additional arguments to pass to the scikit-learn PCA class.

    Attributes:
    - cov_estimator: Robust covariance estimator used for PCA.

    Methods:
    - fit(X, y=None): Fit the PCA model to the training data.
    - transform(X): Apply dimensionality reduction to X.
    - fit_transform(X, y=None): Fit the PCA model to the training data and apply dimensionality reduction to X.

    See the scikit-learn PCA documentation for more details on the inherited methods and attributes.
    """

    def __init__(self, n_components=None, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.cov_estimator = MinCovDet()

    def fit(self, X, y=None):
        """
        Fit the PCA model to the training data.

        Parameters:
        - X: Training data.
        - y: Target labels (default: None).

        Returns:
        - self
        """
        self.cov_estimator.fit(X)
        self.cov_matrix_ = self.cov_estimator.covariance_
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix_)
        sort_indices = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, sort_indices[:self.n_components]]
        self.explained_variance_ = eigenvalues[sort_indices[:self.n_components]]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters:
        - X: Data to transform.

        Returns:
        - Transformed data.
        """
        X_centered = X - self.mean_
        X_transformed = np.dot(X_centered, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit the PCA model to the training data and apply dimensionality reduction to X.

        Parameters:
        - X: Training data.
        - y: Target labels (default: None).

        Returns:
        - Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)    

class OASImpl(OAS):
    """
    Wrapper class for OAS from sklearn.covariance.

    This class inherits from OAS and adds the transform method.

    Methods:
    - transform(X): Apply the Mahalanobis distance transformation to the data.
    """

    def __init__(self, **args):
        super().__init__(**args)
    
    def transform(self, X):
        """
        Apply the Mahalanobis distance transformation to the data.

        Parameters:
        - X: Data to be transformed.

        Returns:
        - Transformed data.
        """
        return self.mahalanobis(X).reshape(-1, 1)**0.5
        
class LedoitWolfImpl(LedoitWolf):
    """
    Wrapper class for LedoitWolf from sklearn.covariance.

    This class inherits from LedoitWolf and adds the transform method.

    Methods:
    - transform(X): Apply the Mahalanobis distance transformation to the data.
    """

    def __init__(self, **args):
        super().__init__(**args)
    
    def transform(self, X):
        """
        Apply the Mahalanobis distance transformation to the data.

        Parameters:
        - X: Data to be transformed.

        Returns:
        - Transformed data.
        """
        return self.mahalanobis(X).reshape(-1, 1)**0.5


class MinCovDetImpl(MinCovDet):
    """
    Wrapper class for MinCovDet from sklearn.covariance.

    This class inherits from MinCovDet and adds the transform method.

    Methods:
    - transform(X): Apply the Mahalanobis distance transformation to the data.
    """

    def __init__(self, **args):
        super().__init__(**args)
    
    def transform(self, X):
        """
        Apply the Mahalanobis distance transformation to the data.

        Parameters:
        - X: Data to be transformed.

        Returns:
        - Transformed data.
        """
        return self.mahalanobis(X).reshape(-1, 1)**0.5

class EllipticEnvelopeImpl(EllipticEnvelope):
    """
    Wrapper class for EllipticEnvelope from sklearn.covariance.

    This class inherits from EllipticEnvelope and adds the transform method.

    Methods:
    - transform(X): Apply the Mahalanobis distance transformation to the data.
    """

    def __init__(self, **args):
        super().__init__(**args)
    
    def transform(self, X):
        """
        Apply the Mahalanobis distance transformation to the data.

        Parameters:
        - X: Data to be transformed.

        Returns:
        - Transformed data.
        """
        return self.mahalanobis(X).reshape(-1, 1)**0.5
        
class NoveltyDetector:
    """
    Novelty detector based on OAS estimation.

    This class provides functionality to fit the detector to training data and
    predict/transform test data.

    Attributes:
    - _pipeline: Data processing pipeline.

    Methods:
    - fit(X, Y=None): Fit the novelty detector to the training data.
    - predict(X, **args): Predict novelty scores for test data.
    - transform(X): Transform test data.
    - copy(): Create a deep copy of the NoveltyDetector object.
    """

    def __init__(self, normalize = False):
        self._pipeline = None
        self._normalize = normalize
        
    def copy(self):
        """
        Create a deep copy of the NoveltyDetector object.

        Returns:
        - Copy of the NoveltyDetector object.
        """
        return copy.deepcopy(self)

    def fit(self, X, Y=None):
        """
        Fit the novelty detector to the training data.

        Parameters:
        - X: Training data.
        - Y: Target labels or values (default: None).

        Returns:
        - self
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self._normalize:
            self._pipeline = make_pipeline(SimpleImputer(strategy='median'), RobustScaler(), OASImpl(), QuantileTransformer(output_distribution = 'normal'))
        else:
            self._pipeline = make_pipeline(SimpleImputer(strategy='median'), RobustScaler(), OASImpl())
        self._pipeline.fit(X, Y)
        return self

    def predict(self, X, **args):
        """
        Predict novelty scores for test data.

        Parameters:
        - X: Test data.

        Returns:
        - Predicted novelty scores.
        """
        return self.transform(X)

    def transform(self, X):
        """
        Transform test data.

        Parameters:
        - X: Test data.

        Returns:
        - Transformed test data.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self._pipeline.transform(X)
    
class InferenceNoveltyDetector:
    """
    Inference-based novelty detector.

    This class provides functionality to fit the detector to training data and
    perform inference on test data.

    Attributes:
    - _model: Underlying machine learning model used for inference.
    - _pipeline: Data processing pipeline.

    Methods:
    - copy(): Create a deep copy of the InferenceNoveltyDetector object.
    - _fit_classifier(X, Y): Fit the classifier-based novelty detector to the training data.
    - _fit_regressor(X, Y): Fit the regressor-based novelty detector to the training data.
    - fit(X, Y): Fit the novelty detector to the training data.
    - predict(X, **args): Perform inference and predict labels or values for test data.
    - predict_proba(X, **args): Perform inference and predict class probabilities for test data.
    - transform(X): Transform test data.
    """

    def __init__(self, model, normalize = True):
        """
        Initialize the InferenceNoveltyDetector.

        Parameters:
        - model: Underlying machine learning model used for inference.
        """
        self._model = model
        self._normalize = normalize
        self._pipeline = []

    def copy(self):
        """
        Create a deep copy of the InferenceNoveltyDetector object.

        Returns:
        - Copy of the InferenceNoveltyDetector object.
        """
        return copy.deepcopy(self)

    def _fit_classifier(self, X, Y):
        """
        Fit the classifier-based novelty detector to the training data.

        Parameters:
        - X: Training data.
        - Y: Target labels.

        Returns:
        - Data processing pipeline.
        """
        if len(np.unique(Y)) == 1:
            constant = np.unique(Y)[0]
            model = DummyClassifierImpl(strategy='constant', constant=constant)
            self._model.model = model

        pipeline = [make_pipeline(SimpleImputer(strategy='median'), RobustScaler()).fit(X)]
        
        imputed = pipeline[0].transform(X)
        pipeline.append(self._model.fit(imputed, Y))
        if self._normalize:
            predicted = pipeline[1].predict_proba(imputed)[:, 1]
            if len(np.unique(predicted)) > 1:
                pipeline.append(QuantileTransformer(output_distribution='normal').fit(predicted.reshape(-1, 1)))
            else:
                pipeline.append(RobustScaler().fit(predicted.reshape(-1, 1)))
        	#pipeline.append(QuantileTransformer(output_distribution='normal').fit(pipeline[1].predict_proba(imputed)[:, 1].reshape(-1, 1)))
        return pipeline

    def _fit_regressor(self, X, Y):
        """
        Fit the regressor-based novelty detector to the training data.

        Parameters:
        - X: Training data.
        - Y: Target values.

        Returns:
        - Data processing pipeline.
        """
        pipeline = [make_pipeline(SimpleImputer(strategy='median'), RobustScaler()).fit(X)]
        
        imputed = pipeline[0].transform(X)
        pipeline.append(self._model.fit(imputed, Y))
        if self._normalize:
            #pipeline.append(QuantileTransformer(output_distribution='normal').fit(pipeline[1].predict(imputed).reshape(-1, 1)))
            
            predicted = pipeline[1].predict(imputed)
            if len(np.unique(predicted)) > 1:
                pipeline.append(QuantileTransformer(output_distribution='normal').fit(predicted.reshape(-1, 1)))
            else:
                pipeline.append(RobustScaler().fit(predicted.reshape(-1, 1)))
            
        return pipeline

    def fit(self, X, Y):
        """
        Fit the novelty detector to the training data.

        Parameters:
        - X: Training data.
        - Y: Target labels or values.

        Returns:
        - self
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if hasattr(self._model, 'predict_proba'):
            self._pipeline = self._fit_classifier(X, Y)
        else:
            self._pipeline = self._fit_regressor(X, Y)

        return self

    def predict(self, X, **args):
        """
        Perform inference and predict labels or values for test data.

        Parameters:
        - X: Test data.
        - **args: Additional arguments to the predict method of the underlying model.

        Returns:
        - Predicted labels or values.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self._pipeline[1].predict(self._pipeline[0].transform(X), **args)

    def predict_proba(self, X, **args):
        """
        Perform inference and predict class probabilities for test data.

        Parameters:
        - X: Test data.
        - **args: Additional arguments to the predict_proba method of the underlying model.

        Returns:
        - Predicted class probabilities.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self._pipeline[1].predict_proba(self._pipeline[0].transform(X), **args)

    def transform(self, X):
        """
        Transform test data.

        Parameters:
        - X: Test data.

        Returns:
        - Transformed test data.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self._normalize:
        	return self._pipeline[2].transform(self._pipeline[1].predict(self._pipeline[0].transform(X)).reshape(-1, 1))
        else:
            return self._pipeline[1].predict(self._pipeline[0].transform(X)).reshape(-1, 1)
