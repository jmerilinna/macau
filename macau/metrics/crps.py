#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import properscoring as ps

from scipy.stats import norm

class CRPS:
    def __init__(self):
        """
        Continuous Ranked Probability Score (CRPS) is a scoring function for Bayesian machine learning models.
        Based on Tilmann Gneiting, Adrian E. Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation'.
        """
        pass
    
    def _crps(self, y_true, y_pred):
        """
        Calculate the CRPS for a single set of true and predicted values.
        
        Args:
        - y_true: 1D array or list, true values
        - y_pred: 1D array or list, predicted values
        
        Returns:
        - float, CRPS value
        """
        n_preds = y_pred.shape[0]
        absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)
        
        if n_preds == 1:
            return np.median(absolute_error)
        
        y_pred = np.sort(y_pred, axis=0)
        diff = np.diff(y_pred, axis=0)
        weight = (np.arange(1, n_preds) * np.arange(1, n_preds)[::-1]).reshape(-1, 1)
        return np.median(absolute_error - np.sum(diff * weight, axis=0) / n_preds**2)
    
    def _empirical_crps(self, y_true, y_pred):
        return ps.crps_ensemble(y_true, y_pred)
        
    def ecrps(self, y_true, y_pred):    
        """
        Calculate the empirical CRPS for a set of true and predicted values.
        
        Args:
        - y_true: 1D array or list, true values
        - y_pred: 2D array or list, predicted values
        
        Returns:
        - float, empirical CRPS value
        """
        if isinstance(y_true, np.ndarray) or isinstance(y_true, list):
            #return np.median([self._crps(y, y_pred[idx]) for idx, y in enumerate(y_true)])
            return np.mean([self._empirical_crps(y, y_pred[idx]) for idx, y in enumerate(y_true)])
        else:
            return self._crps(y_true, y_pred)
        return self._empirical_crps(y_true, y_pred)
    
    def crps(self, y_true, y_mean, y_sigma, n=100):            
        """
        Calculate the CRPS for a set of true values and predicted mean/sigma values.
        
        Args:
        - y_true: 1D array or list of true values or single true value
        - y_mean: 1D array or list of predicted mean values, or single predicted value
        - y_sigma: 1D array or list of predicted sigma (standard deviation) values, or single sigma value
        - n: int, number of points for approximating the predictive distribution using Percent Point Function (PPF)
        
        Returns:
        - float, CRPS value
        """
        ppf = norm.ppf(np.linspace(0.025, 0.975, n))
        if isinstance(y_true, np.ndarray) or isinstance(y_true, list):   
            
            #return np.median([self._crps(y, y_mean[idx] + y_sigma[idx] * ppf) for idx, y in enumerate(y_true)])
            return np.mean([self._empirical_crps(y, y_mean[idx] + y_sigma[idx] * ppf) for idx, y in enumerate(y_true)])
        else:
            #return self._crps(y_true, y_mean + y_sigma * ppf)
            return self._empirical_crps(y_true, y_mean + y_sigma * ppf)
