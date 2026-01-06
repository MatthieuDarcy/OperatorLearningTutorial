

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class GlobalStandardScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.global_mean_ = np.mean(X)
        self.global_std_ = np.std(X)
        return self

    def transform(self, X):
        return (X - self.global_mean_) / self.global_std_

    def inverse_transform(self, X_scaled):
        return X_scaled * self.global_std_ + self.global_mean_
    
    

#%% score using the relative L2 loss

from sklearn.metrics import make_scorer

def relative_l2_loss(y_true, y_pred):
    """
    Computes the relative L2 loss for each sample in a batch (vectorized).

    Parameters:
    y_true: ndarray of shape (N, ...)
    y_pred: ndarray of shape (N, ...)

    Returns:
    mean relative L2 loss over N samples
    """
    # diffs = y_true - y_pred
    # num = np.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)
    # denom = np.linalg.norm(y_true.reshape(y_true.shape[0], -1), axis=1)
    # rel_l2 = num / denom

    rel_l2 = itemize_relative_l2_loss(y_true, y_pred)
    assert rel_l2.ndim == 1, "rel_l2 must be a 1D array"
    assert rel_l2.shape[0] == y_true.shape[0], "rel_l2 must have the same number of samples as y_true"

    return np.mean(rel_l2)

def itemize_relative_l2_loss(y_true, y_pred):
    """
    Computes the relative L2 loss for each sample in a batch (vectorized).

    Parameters:
    y_true: ndarray of shape (N, ...)
    y_pred: ndarray of shape (N, ...)

    Returns:
    relative L2 loss for each sample
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    N = y_true.shape[0]
    diffs = y_true - y_pred
    num = np.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)
    denom = np.linalg.norm(y_true.reshape(y_true.shape[0], -1), axis=1)
    assert num.shape == denom.shape == (N,), "num and denom must have the same shape"
    return num / denom

def l2_loss(y_true, y_pred):
    """
    Computes the L2 loss for each sample in a batch (vectorized).

    Parameters:
    y_true: ndarray of shape (N, ...)
    y_pred: ndarray of shape (N, ...)

    Returns:
    mean L2 loss over N samples
    """
    diffs = y_true - y_pred
    num = np.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)
    return np.mean(num)

relative_l2_scorer = make_scorer(relative_l2_loss, greater_is_better=False)

#%% Define the GPR class that uses the pipelines

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,  ConstantKernel, DotProduct
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

class BenchmarkGPR(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 pca_input_components=0, 
                 alpha=1e-10, 
                 length_scale=1.0,  
                 pca_output_components=0,
                 normalize=True,
                 weight_matern=0.5):
        self.pca_input_components = pca_input_components
        self.alpha = alpha
        self.length_scale = length_scale
        self.normalize = normalize
        self.pca_output_components = pca_output_components
        self.weight_matern = weight_matern
        self.weight_dot = 1- weight_matern

        self.pca_input = pca_input_components
        self.pca_output = pca_output_components

        self._create_pipeline()

    def _create_pipeline(self):

        kernel = ConstantKernel(constant_value=self.weight_matern, constant_value_bounds= 'fixed')* Matern(length_scale=self.length_scale, length_scale_bounds="fixed") \
            + ConstantKernel(constant_value=self.weight_dot, constant_value_bounds= 'fixed')* DotProduct(sigma_0_bounds = 'fixed')
        
        if self.normalize and self.pca_input_components > 0:
            self.pipeline = Pipeline([
                ("scaler", GlobalStandardScaler()),
                ("pca", PCA(n_components=self.pca_input_components)),
                ("gpr", GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, normalize_y=False))
            ])
        elif self.normalize and self.pca_input_components == 0:
            self.pipeline = Pipeline([
                ("scaler", GlobalStandardScaler()),
                ("gpr", GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, normalize_y=False))
            ])
        elif self.pca_input_components > 0 and not self.normalize:
            self.pipeline = Pipeline([
                ("pca", PCA(n_components=self.pca_input_components)),
                ("gpr", GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, normalize_y=False))
            ])
        else:
            self.pipeline = Pipeline([
                ("gpr", GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, normalize_y=False))
            ])

        if self.normalize and self.pca_output_components > 0:
            self.output_pipeline = Pipeline([
                ("scaler", GlobalStandardScaler()),
                ("pca", PCA(n_components=self.pca_output_components))
            ])
        elif self.normalize and self.pca_output_components == 0:
            self.output_pipeline = Pipeline([
                ("scaler", GlobalStandardScaler())
            ])
        elif self.pca_output_components > 0 and not self.normalize:
            self.output_pipeline = Pipeline([
                ("pca", PCA(n_components=self.pca_output_components))
            ])
        else:
            self.output_pipeline = None

    def fit(self, X, y):
        if self.output_pipeline is not None:
            y = self.output_pipeline.fit_transform(y)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        pred = self.pipeline.predict(X)
        if self.output_pipeline is not None:
            if len(pred.shape) == 1:
                pred = pred[:, None] # add a dimension if it's a single output
            pred = self.output_pipeline.inverse_transform(pred)
        return pred
    
    def score(self, X, y):
        # The score is the relative L2 loss
        pred = self.predict(X)
        return relative_l2_loss(y, pred)