import os
import numpy as np
import pandas
from sklearn.linear_model import ARDRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from tqdm import tqdm
from joblib import Parallel, delayed
import lightgbm
from macau.model.linear_model import DummyClassifierCI, DummyRegressorCI, DeltaLogisticRegression
from macau.novelty.noveltydetector import NoveltyDetector, InferenceNoveltyDetector

import numpy as np
import scipy
from scipy.linalg import norm
import copy
from sklearn.covariance import LedoitWolf, MinCovDet, EllipticEnvelope, OAS
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from macau.model.linear_model import DummyClassifierImpl

    
class MACAU:
    def __init__(self, model,
                 linear_tree = True):
        """
        Conducts tree inspection of LightGBMClassifier and LightGBMRegressor to fit multiple models to the leaves of the trees to enable uncertainty modelling.

        Parameters
        ----------
        model : Fitted Lightgbm model
        linear_tree : bool, optional
            True: Use piece-wise linear tree.
            False: Use leaf-wise mean for predictions.
            (default is True)
        
        Usage
        -----
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        import lightgbm
        X, Y = load_wine(return_X_y = True, as_frame = False)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
        model = lightgbm.LGBMClassifier(boosting_type = 'rf',
                                        n_estimators = 100,
                                        max_depth = -1,
                                        subsample_freq = 1,
                                        subsample = 0.8)
        model.fit(X_train, Y_train)
        macau = MACAU(model)).fit(X_train, Y_train)
        result = macau.predict(X_test)
        """

        self._model = model
        self._estimators = []

        self._linear_tree = linear_tree
        self._classes = [0]
        
    @property
    def is_classifier(self):
        return isinstance(self._model, lightgbm.LGBMClassifier)

    def _df_to_ndarray(self, X):
        if isinstance(X, pandas.DataFrame):
            return X.values
        return X

    def _series_to_array(self, Y):
        if isinstance(Y, pandas.Series):
            return Y.values
        return Y

    def _get_selected_features(self, tree, tree_index, leaf_index):
        features = set()
        current_tree = tree[tree['tree_index'] == tree_index]
        current_node = f'{tree_index}-L{leaf_index}'
        node_info = current_tree[current_tree['node_index'] == current_node].iloc[0]
        n_samples = node_info['count']
        current_node = node_info['parent_index']
        while current_node is not None:
            node_info = current_tree[current_tree['node_index'] == current_node].iloc[0]
            split_feature = node_info['split_feature']
            if split_feature is not None:
                features.add(split_feature)
            current_node = node_info['parent_index']
        return list(features), n_samples

    def _feature_names_to_index(self, model, features):
        return sorted([model.feature_name_.index(feature) for feature in features])

    def _get_active_features(self, tree, estimator_idx, unique_leaf):
        active_features, n_samples = self._get_selected_features(tree, estimator_idx, unique_leaf)
        active_features = self._feature_names_to_index(self._model, active_features)
        return active_features, n_samples

    def _fit_regression_estimators(self, X, Y, leaves, tree, estimator_idx):
        estimators = {}  # Dictionary to store the fitted estimators for each unique leaf
        
        # Iterate over unique leaves in the provided 'leaves' array
        for unique_leaf in np.unique(leaves, axis=0):
            feats, n_samples = self._get_active_features(tree, estimator_idx, unique_leaf)
            if not feats:
                feats = list(range(X.shape[1]))  # Use all features if none are active in the leaf
    
            # Initialize the estimator dictionary for the current leaf
            estimators[unique_leaf] = {'novelty_estimator': None,
                                       'inference_novelty_estimator': None,
                                       'conditional_novelty_estimator': None,
                                       'uncertainty_estimator': None,
                                       'active_features': None,
                                       'n_samples': 0}
    
            selected_Y = Y[leaves == unique_leaf]  # Select the target values corresponding to the current leaf
            selected_X = X[leaves == unique_leaf]  # Select the feature values corresponding to the current leaf
    
            estimators[unique_leaf]['n_samples'] = selected_X.shape[0]  # Store the number of samples in the leaf
            estimators[unique_leaf]['active_features'] = feats  # Store the active features for the leaf
    
            # Fit the novelty estimators for the current leaf
            estimators[unique_leaf]['novelty_estimator'] = NoveltyDetector().fit(selected_X, selected_Y)
            if len(feats) == selected_X.shape[1]:
                estimators[unique_leaf]['conditional_novelty_estimator'] = estimators[unique_leaf]['novelty_estimator']
            else:
                estimators[unique_leaf]['conditional_novelty_estimator'] = NoveltyDetector().fit(selected_X[:, feats], selected_Y)
            
            # Fit the uncertainty estimator for the leaf
            if not self._linear_tree:
                # If linear tree is disabled, fit a DummyRegressorCI to estimate uncertainties
                estimators[unique_leaf]['uncertainty_estimator'] = DummyRegressorCI().fit(selected_X[:, estimators[unique_leaf]['active_features']], np.mean(selected_Y) * np.ones(len(selected_Y)), std=np.std(selected_Y))
                estimators[unique_leaf]['inference_novelty_estimator'] = InferenceNoveltyDetector(DummyRegressorCI(), normalize = True).fit(selected_X[:, estimators[unique_leaf]['active_features']], np.mean(selected_Y) * np.ones(len(selected_Y)))
            else:
                estimators[unique_leaf]['inference_novelty_estimator'] = InferenceNoveltyDetector(ARDRegression(), normalize = True).fit(selected_X[:, feats], selected_Y)
                estimators[unique_leaf]['uncertainty_estimator'] = estimators[unique_leaf]['inference_novelty_estimator']

        return estimators

    def _fit_classifier_estimators(self, X, Y, leaves, tree, estimator_idx):
        # Compute unique labels and their counts in the target variable
        labels, label_counts = np.unique(Y, return_counts=True)
        
        # Determine the label to be used for the current estimator
        if len(labels) > 2:
            estimator_label = labels[estimator_idx % len(labels)]
        else:
            estimator_label = labels[-1]
    
        # Initialize dictionary to store estimators for each leaf
        estimators = {}
        
        # Iterate over unique leaves
        for unique_leaf in np.unique(leaves, axis=0):
            # Get the active features and the number of samples in the current leaf
            feats, n_samples = self._get_active_features(tree, estimator_idx, unique_leaf)
            
            # If no active features are found, consider all features
            if not feats:
                feats = list(range(X.shape[1]))
    
            # Create a dictionary to store estimators for the current leaf
            estimators[unique_leaf] = {'novelty_estimator': None,
                                       'inference_novelty_estimator': None,
                                       'conditional_novelty_estimator': None,
                                       'uncertainty_estimator': None,
                                       'active_features': None,
                                       'n_samples': 0}
    
            # Select the samples and features for the current leaf
            selected_Y = Y[leaves == unique_leaf]
            selected_X = X[leaves == unique_leaf]
    
            # Set the number of samples and active features in the estimators dictionary
            estimators[unique_leaf]['n_samples'] = selected_X.shape[0]
            estimators[unique_leaf]['active_features'] = feats
    
            # Fit novelty detectors for conditional and unconditional novelties
            estimators[unique_leaf]['novelty_estimator'] = NoveltyDetector().fit(selected_X, selected_Y)
            if len(feats) == selected_X.shape[1]:
                estimators[unique_leaf]['conditional_novelty_estimator'] = estimators[unique_leaf]['novelty_estimator']
            else:
                estimators[unique_leaf]['conditional_novelty_estimator'] = NoveltyDetector().fit(selected_X[:, feats], selected_Y)
            
            # Determine the uncertainty estimator based on the linear tree option
            if not self._linear_tree:
                # Use a constant dummy classifier as the uncertainty estimator
                estimators[unique_leaf]['inference_novelty_estimator'] = InferenceNoveltyDetector(DummyClassifierCI(constant=np.mean((selected_Y == estimator_label).astype(int)), std=np.std((selected_Y == estimator_label).astype(int))), normalize = True).fit(selected_X[:, feats], (selected_Y == estimator_label).astype(int))
            else:
                # Use the inference novelty estimator as the uncertainty estimator
                estimators[unique_leaf]['inference_novelty_estimator'] = InferenceNoveltyDetector(DeltaLogisticRegression(), normalize = True).fit(selected_X[:, feats], (selected_Y == estimator_label).astype(int))
            estimators[unique_leaf]['uncertainty_estimator'] = estimators[unique_leaf]['inference_novelty_estimator']
                
        return estimators

    def _fit_estimators(self, X, Y, leaves, tree, estimator_idx):
        if self.is_classifier:
            return self._fit_classifier_estimators(X, Y, leaves, tree, estimator_idx)
        else:
            return self._fit_regression_estimators(X, Y, leaves, tree, estimator_idx)

    def _fit(self, X, Y, leaves, n_jobs=-1, verbose=0):
        """
        Fit the estimators using the provided data and leaves.
    
        Parameters:
            X (array-like): Input data.
            Y (array-like): Target labels.
            leaves (numpy.ndarray): Leaves array indicating which leaf each sample belongs to.
            n_jobs (int, optional): Number of parallel jobs. Default is -1, which uses all available CPUs.
            verbose (int, optional): Verbosity level. Default is 0.
    
        Returns:
            self: The fitted model instance.
    
        Notes:
            - This function fits the estimators to the leaves of the model's trees.
            - The `_fit_estimators` method is called in parallel for each estimator index.
    
        """
        if verbose > 1:
            verbose = 51
    
        if n_jobs <= 0:
            n_jobs = os.cpu_count()
        if n_jobs > 1:
            backend = 'loky'
        else:
            backend = 'threading'
    
        tree_df = self._model._Booster.trees_to_dataframe()
    
        jobs = (delayed(self._fit_estimators)(self._df_to_ndarray(X), self._series_to_array(Y), leaves[:, estimator_idx], tree_df, estimator_idx) for estimator_idx in range(leaves.shape[1]))
        result = Parallel(n_jobs=n_jobs, backend=backend, verbose=np.clip(verbose - 1, 0, 50))(jobs)
        for estimator_idx in range(leaves.shape[1]):
            self._estimators.append(result[estimator_idx])
    
        return self

    def _predict(self, X, leaves, estimators):
        """
        Predict using the fitted estimators.
    
        Parameters:
            X (array-like): Input data.
            leaves (numpy.ndarray): Leaves array indicating which leaf each sample belongs to.
            estimators (dict): Dictionary containing the fitted estimators for each leaf.
    
        Returns:
            list: A list containing the predictions, novelty scores, conditional novelty scores,
                  uncertainty scores, and number of samples for each input sample.
    
        Notes:
            - This function iterates over unique leaves and retrieves the corresponding estimators.
            - Predictions, inference novelty scores, novelty scores, conditional novelty scores, and uncertainty scores are computed for each sample.
            - If an estimator is not available for a leaf, NaN values are assigned.
            - For classifiers, predictions are probabilities of the positive class.
            - The function returns a list containing the computed scores and predictions for each sample.
    
        """
        predictions = np.nan * np.zeros(X.shape[0])
        novelty = np.zeros(X.shape[0])
        conditional_novelty = np.zeros(X.shape[0])
        inference_novelty = np.zeros(X.shape[0])
        uncertainty = np.zeros(X.shape[0])
        n_samples = np.zeros(X.shape[0])
    
        for unique_leaf in np.unique(leaves):
            matches = leaves == unique_leaf
            selected_X = X[matches]
            n_samples[matches] = estimators[unique_leaf]['n_samples']
            
            if estimators[unique_leaf]['novelty_estimator'] is None:
                novelty[matches] = np.nan
            else:
                novelty[matches] = estimators[unique_leaf]['novelty_estimator'].transform(selected_X).ravel()
                conditional_novelty[matches] = estimators[unique_leaf]['conditional_novelty_estimator'].transform(selected_X[:, estimators[unique_leaf]['active_features']]).ravel()
            
            if estimators[unique_leaf]['inference_novelty_estimator'] is None:
                inference_novelty[matches] = np.nan
            else:
                inference_novelty[matches] = estimators[unique_leaf]['inference_novelty_estimator'].transform(
                    selected_X[:, estimators[unique_leaf]['active_features']]
                ).ravel()
            
            if estimators[unique_leaf]['uncertainty_estimator'] is None:
                uncertainty[matches] = np.nan
            else:
                if self.is_classifier:
                    preds, std = estimators[unique_leaf]['uncertainty_estimator'].predict_proba(selected_X[:, estimators[unique_leaf]['active_features']], return_std=True)
                    preds = preds[:, 1]
                else:
                    preds, std = estimators[unique_leaf]['uncertainty_estimator'].predict(selected_X[:, estimators[unique_leaf]['active_features']], return_std=True)
                predictions[matches] = preds
                uncertainty[matches] = std
        
        return [predictions, inference_novelty, novelty, conditional_novelty, uncertainty, n_samples]

    
    def _combine_estimations(self, data, n_classes, n_samples):
        """
        Combine the results from each tree into a single combined result.
    
        Parameters:
            data (numpy.ndarray): Array containing the individual tree results.
            n_classes (int): Number of classes in the classification problem.
            n_samples (numpy.ndarray): Array containing the number of samples in each leaf.
    
        Returns:
            numpy.ndarray: Array containing the combined estimations for each sample.
    
        Notes:
            - This function combines the results of each tree to obtain a single estimation for each sample.
            - For multi-class problems (n_classes > 2), the estimations are computed separately for each class.
            - Weighted mean or median is used to compute the means of the estimations.
            - Standard deviations are computed using either sample standard deviation or combined standard deviation.
            - The function returns an array containing the combined estimations for each sample.
        """
    
        if n_classes is None or n_classes > 2:
            estimations = []
            n_estimators = data.shape[2] // n_classes
            for class_id in range(n_classes):
                # Compute means using weighted mean or median
                means = [np.mean(data[:, i, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)], axis=1) for i in range(data.shape[1] - 1)]
    
                # Compute novelty standard deviations using weighted sample standard deviation
                inference_novelty_stds = np.average(
                    (data[:, 1, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)] - means[1].reshape(-1, 1))**2,
                    weights=n_samples[:, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)] - 1,
                    axis=1
                )**0.5
                
                novelty_stds = np.average(
                    (data[:, 2, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)] - means[2].reshape(-1, 1))**2,
                    weights=n_samples[:, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)] - 1,
                    axis=1
                )**0.5
    
                # Compute conditional novelty standard deviations using weighted sample standard deviation
                conditional_novelty_stds = np.average(
                    (data[:, 3, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)] - means[3].reshape(-1, 1))**2,
                    weights=n_samples[:, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)] - 1,
                    axis=1
                )**0.5
    
                # Compute epistemic uncertainty, aleatoric uncertainty, and combined uncertainty
                epistemic_uncertainty = np.std(data[:, 0, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)], axis=1)
                aleatoric_uncertainty = np.mean(data[:, -1, np.arange(class_id, n_estimators * n_classes, n_classes).astype(int)], axis=1)
    
                # Stack the computed results and append to the estimations list
                result = np.vstack([means, inference_novelty_stds, novelty_stds, conditional_novelty_stds, aleatoric_uncertainty, epistemic_uncertainty, aleatoric_uncertainty + epistemic_uncertainty]).T
                estimations.append(result)
    
            # Convert estimations list to a numpy array and transpose the dimensions
            return np.array(estimations)
        else:
            # Compute means using weighted mean or median
            means = [np.mean(data[:, i], axis=1) for i in range(data.shape[1] - 1)]
    
            # Compute epistemic uncertainty, aleatoric uncertainty, novelty standard deviations, and conditional novelty standard deviations
            epistemic_uncertainty = np.std(data[:, 0], axis=1)
            aleatoric_uncertainty = np.mean(data[:, -1], axis=1)
            
            inference_novelty_stds = np.average((data[:, 1] - means[1].reshape(-1, 1))**2, weights=n_samples - 1, axis=1)**0.5
            novelty_stds = np.average((data[:, 2] - means[2].reshape(-1, 1))**2, weights=n_samples - 1, axis=1)**0.5
            conditional_novelty_stds = np.average((data[:, 3] - means[3].reshape(-1, 1))**2, weights=n_samples - 1, axis=1)**0.5
    
            # Stack the computed results
            return np.vstack([means, inference_novelty_stds, novelty_stds, conditional_novelty_stds, aleatoric_uncertainty, epistemic_uncertainty, aleatoric_uncertainty + epistemic_uncertainty]).T

    
    def predict(self, X, n_jobs=1, verbose=0):
        """
        Predict the output for the given input samples.
    
        Parameters:
            X: Input samples to be predicted.
            n_jobs (int): Number of parallel jobs to run. Default is 1.
            verbose (int): Verbosity level. Default is 0.
    
        Returns:
            numpy.ndarray or list or pandas.DataFrame:
                - If the input is a numpy array or pandas DataFrame and the problem is regression or binary classification,
                  it returns a numpy array of shape (n_samples, 10) containing the predictions and uncertainties.
                - If the input is a numpy array or pandas DataFrame and the problem is multi-class classification,
                  it returns a numpy array of shape (n_classes, n_samples, 10) containing the predictions and uncertainties
                  for each class separately.
                - If the input is a pandas DataFrame, the function returns a DataFrame as a result(s), where each column
                  represents the predictions and uncertainties for each sample.
                  In multi-class classification, the column names are appended with '_class_{class}'.
    
        Notes:
            - This function predicts the output for the given input samples based on the trained model.
            - The output includes predictions, novelty measures, conditional novelty measures, novelty standard deviation,
              conditional novelty standard deviation, aleatoric uncertainty, epistemic uncertainty, and total uncertainty.
            - The number of returned columns is always 8.
            - The result depends on the problem type and the input type.
            - If the input is a pandas DataFrame, the result is also a DataFrame with appropriate column names.
            - The function supports parallel execution using multiple threads.
            - The verbosity level controls the amount of logging information displayed during prediction.
        """
        
        if verbose > 1:
            verbose = 51
    
        if n_jobs <= 0:
            n_jobs = os.cpu_count()
 
        leaves = self._model.predict(X, pred_leaf=True)
        if len(leaves.shape) == 1:
            leaves = leaves.reshape(-1, 1)
    
        jobs = (delayed(self._predict)(self._df_to_ndarray(X), leaves[:, estimator_idx], self._estimators[estimator_idx]) for estimator_idx in range(leaves.shape[1]))
        result = Parallel(n_jobs=n_jobs, backend='threading', verbose=np.clip(verbose - 1, 0, 50))(jobs)
        result = np.array(result).T
        result = self._combine_estimations(result[:, :-1], n_classes=len(self._classes), n_samples=result[:, -1])
    
        if isinstance(X, pandas.DataFrame):
            columns = ['prediction', 'inference_novelty', 'novelty', 'conditional_novelty', 'inference_novelty_uncertainty', 'novelty_uncertainty', 'conditional_novelty_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty', 'uncertainty']
            if len(result.shape) == 2:
                return pandas.DataFrame(result, columns=columns, index=X.index)
            else:
                return [pandas.DataFrame(result[idx], columns=[f'{col}_class_{self._classes[idx]}' for col in columns], index=X.index) for idx in range(result.shape[0])]
        return result


    def fit(self, X, Y, n_jobs=-1, verbose=0):
        """
        Fits the MACAU model to the provided training data.
    
        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input training samples.
    
        Y : array-like, shape (n_samples,)
            The target values or labels for classification tasks.
    
        n_jobs : int, optional
            The number of parallel jobs to use for fitting estimators. 
            Set to -1 to use all available CPU cores (default).
    
        verbose : int, optional
            Verbosity level. Set to 0 for no output, 1 to show progress bar, 
            and 2 to show detailed logging.
    
        Returns:
        --------
        self : object
            Returns the instance of the MACAU model after fitting.
    
        Notes:
        ------
        - This function trains the MACAU model using the provided input data.
        - The MACAU model uses LightGBM random forest booster and fits estimators to the
          leaves of the underlying model.
        - The training progress can be visualized using the progress bar if verbose is set to 1.
    
        """
        if verbose > 1:
            verbose = 51
    
        pbar = tqdm(total=2, desc='Predicting leaves', disable=verbose < 1)
    
        self._estimators = []
        self._classes = [0]
    
        leaves = self._model.predict(X, pred_leaf=True)
        if len(leaves.shape) == 1:
            leaves = leaves.reshape(-1, 1)
        if self.is_classifier:
            self._classes = np.unique(Y)
    
        pbar.update(1)
        pbar.set_description('Fitting estimators')
        self._fit(X, Y, leaves, n_jobs, verbose)
        pbar.update(1)
        return self
