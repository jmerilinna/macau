import numpy as np
from scipy.stats import chi2

class ShapleyCellDetector:
    """
    This is an implementation of Shapley Cell Detector algorithm
    introduced in:
        
    Mayrhofer, M., Filzmoser, P.,
    Multivariate outlier explanations using Shapley values and Mahalanobis distances,
    URL: https://arxiv.org/abs/2210.10063
    """
    def __init__(self, cov):
        self._cov = cov
        self._location = cov.location_
        self._precision = cov.precision_

    def novelty_contributions(self, X):
        """
        Calculates Shapley novelty values for each feature.
        """
        return ((X - self._location) @ self._precision) * (X - self._location)

    def shapley_cell_detector(self, X, d=0.1, features_to_vary=None, threshold=0.95):
        """
        Implements Shapley Cell Detector algorithm with an option to choose allowed
        features to be varied in the process.
        """
        if features_to_vary is None:
            features_to_vary = np.arange(X.shape[1])

        X_hat = X.copy()
        remaining_rows = np.arange(X_hat.shape[0])
        
        while len(remaining_rows) > 0:
            contrib = self.novelty_contributions(X_hat[remaining_rows])
            novelty_proba = chi2.cdf(np.sum(contrib, axis=1), X.shape[1])
            remaining_rows = remaining_rows[novelty_proba > threshold]
            
            for i in np.arange(len(remaining_rows)):
                j = features_to_vary[np.argmax(contrib[i, features_to_vary])]
                X_hat[remaining_rows[i], j] -= d * (X_hat[remaining_rows[i], j] - self._location[j])
        return X - X_hat
