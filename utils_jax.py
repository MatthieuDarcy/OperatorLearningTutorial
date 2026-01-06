#%%
from jax import numpy as jnp
from jax import jit

from jax import vmap

from jax import scipy

import jax
#%%

class GlobalStandardScaler():
    def fit(self, X, y=None, epsilon = 1e-5):
        """
        X array of shape (N, d) 
        """
        self.episilon = epsilon
        self.global_mean_ = jnp.mean(X)
        self.global_std_ = jnp.std(X)   
        if self.global_std_ < self.episilon:
            self.global_std_ = 1.0
        return self

    def transform(self, X):
        return (X - self.global_mean_) / self.global_std_
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled * self.global_std_ + self.global_mean_
    
#%%

import pcax

#%%

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
    num = jnp.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)
    denom = jnp.linalg.norm(y_true.reshape(y_true.shape[0], -1), axis=1)
    assert num.shape == denom.shape == (N,), "num and denom must have the same shape"
    return num / denom

def relative_l2_loss(y_true, y_pred):
    """
    Computes the relative L2 loss for each sample in a batch (vectorized).

    Parameters:
    y_true: ndarray of shape (N, ...)
    y_pred: ndarray of shape (N, ...)

    Returns:
    mean relative L2 loss over N samples
    """
    rel_l2 = itemize_relative_l2_loss(y_true, y_pred)
    assert rel_l2.ndim == 1, "rel_l2 must be a 1D array"
    assert rel_l2.shape[0] == y_true.shape[0], "rel_l2 must have the same number of samples as y_true"

    return jnp.mean(rel_l2).item()  # Convert to a scalar

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
    num = jnp.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)
    return jnp.mean(num)

def squared_l2_loss(y_true, y_pred):
    """
    Computes the squared L2 loss for each sample in a batch (vectorized).

    Parameters:
    y_true: ndarray of shape (N, ...)
    y_pred: ndarray of shape (N, ...)

    Returns:
    mean squared L2 loss over N samples
    """
    diffs = y_true - y_pred
    num = jnp.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)**2
    return jnp.mean(num)

def itemized_squared_l2_error(y_true, y_pred, eps=1e-8):
    assert y_true.shape == y_pred.shape
    error = jnp.sum(y_true - y_pred)**2
    norm = jnp.sum(y_true)**2
    return error/ (norm+ eps)

def relative_squared_l2_error(y_true, y_pred):
    rel_l2 = jax.vmap(itemized_squared_l2_error)(y_true, y_pred)
    assert rel_l2.ndim == 1, "rel_l2 must be a 1D array"
    return jnp.mean(rel_l2)  # Convert to a scalar

#%% Define a PCA module similar to sklearn's PCA
class PCA():
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        self.state = pcax.fit(X, n_components=self.n_components)
        self.explaine_variance_ = self.state.explained_variance
        self.explained_variance_ratio_ = self.state.explained_variance/ self.state.explained_variance.sum()
        return self

    def transform(self, X):
        return pcax.transform(self.state, X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        return pcax.recover(self.state, X_transformed)
    
    def explaine_variance_(self):
        return self.state.explained_variance
    
    def explained_variance_ratio_(self):
        return self.state.explained_variance_/ self.state.explained_variance_.sum()


#%%

# Defining the second order elliptic operators 

def matern_kernel(x, y, length_scale, nu = "1.5"):
    r = jnp.sqrt(jnp.sum((x - y) ** 2))
    if nu == "1.5":
        return (1 + jnp.sqrt(3)*r/length_scale) * jnp.exp(-jnp.sqrt(3)*r/length_scale)
    elif nu == "2.5": 
        return (1 + jnp.sqrt(5)*r/length_scale + (5 / 3) * (r ** 2) / (length_scale ** 2)) * jnp.exp(-jnp.sqrt(5)*r/length_scale)
    elif nu == "0.5":
        return jnp.exp(-r/length_scale)
    elif nu == "inf":
        return jnp.exp(-r**2/(2*length_scale**2))
    else:
        raise ValueError(f"Unsupported nu value: {nu}. Supported values are 0.5, 1.5, 2.5, and inf.")

def matern_1_5_kernel(x, y, length_scale):
    r = jnp.sqrt(jnp.sum((x - y) ** 2))
    return (1 + jnp.sqrt(3)*r/length_scale) * jnp.exp(-jnp.sqrt(3)*r/length_scale)

def matern_2_5_kernel(x, y, length_scale):
    r = jnp.sqrt(jnp.sum((x - y) ** 2))
    return (1 + jnp.sqrt(5)*r/length_scale + (5 / 3) * (r ** 2) / (length_scale ** 2)) * jnp.exp(-jnp.sqrt(5)*r/length_scale)

def matern_0_5_kernel(x, y, length_scale):
    r = jnp.sqrt(jnp.sum((x - y) ** 2))
    return jnp.exp(-r/length_scale)

def matern_inf_kernel(x, y, length_scale):
    r = jnp.sqrt(jnp.sum((x - y) ** 2))
    return jnp.exp(-r**2/(2*length_scale**2))


def matern_kernel_create(nu="1.5"):
    """
    Creates a matern kernel function with the specified length scale and nu value.
    
    Parameters:
    length_scale: float, the length scale of the kernel
    nu: str, the nu value of the kernel, can be "0.5", "1.5", "2.5", or "inf"
    
    Returns:
    A function that computes the matern kernel between two points.
    """
    if nu == "1.5":
        return matern_1_5_kernel
    elif nu == "2.5":
        return matern_2_5_kernel
    elif nu == "0.5":
        return matern_0_5_kernel
    elif nu == "inf":
        return matern_inf_kernel
    else:
        raise ValueError(f"Unsupported nu value: {nu}. Supported values are 0.5, 1.5, 2.5, and inf.")


def dot_kernel(x, y):
    return jnp.dot(x, y)
# vmap_kernel_row = vmap(matern_kernel, in_axes=(None, 0, None))
# # Now we apply vmap to the result to vectorize over the rows of the first argument
# vmap_kernel = jit(vmap(vmap_kernel_row, in_axes=(0, None, None)))


#%%

class BenchmarkGPR():
    def __init__(self, 
                 kernel,
                 parameters,
                 pca_input_components=0, 
                 alpha=1e-10,   
                 pca_output_components=0,
                 normalize=True,
                 batch_size = None,
                 jit_kernel = True, # recommended to be True if computations will be repeated,
                 save_K = False, # if True, the kernel matrix will be saved in self.K
                 relative_l2= False # wether to use relative L2 loss or not
):
        self.pca_input_components = int(pca_input_components)
        self.alpha = alpha
        self.normalize = normalize
        self.pca_output_components = int(pca_output_components)
        self.pca_input = pca_input_components
        self.pca_output = pca_output_components
        self.parameters = jnp.array(parameters)

        self.kernel_function = kernel
        self.batch_size = batch_size

        self.relative_l2 = relative_l2

        vmap_kernel_row = vmap(kernel, in_axes=(None, 0, None))
        # # Now we apply vmap to the result to vectorize over the rows of the first argument
        if jit_kernel:
            self.kernel = jit(vmap(vmap_kernel_row, in_axes=(0, None, None)))
        else:
            self.kernel= vmap(vmap_kernel_row, in_axes=(0, None, None))

        self.save_K = save_K



    def fit(self, X, y):

        if self.relative_l2:
            # If using the relative L2 loss, the regularization weights are the squared L2 norm of the target values
            # (inverse of the loss weights)
            self.reg_weights = jnp.linalg.norm(y, axis=1)**2 + 1e-11 # Adding a small value to avoid division by zero
        else:
            self.reg_weights = None

        X = self.create_transform_x(X)
        y = self.create_transform_y(y)

        self.fit_kernel(X, y, weights=self.reg_weights)

        return self
    
    def fit_kernel(self, X, y, weights = None):
        """
        X array of shape (N, d)
        y array of shape (N, m)
        weights array of shape (N,) or None: weights for the regularization term 
                                            (equivalent to using a weighted L^2 loss, 
                                            the regularization weights are the inverse of the loss weights)
        """
        self.X = X
        K = self.kernel(X, X, self.parameters)
        if self.save_K:
            self.K = K
        #print(K.shape)
        if weights is not None:
            # If weights are provided, we will use them to compute the weighted kernel matrix
            reg = self.alpha*jnp.diag(weights)
        else:
            reg = self.alpha * jnp.eye(X.shape[0])
        self.L_ = scipy.linalg.cho_factor(K + reg, lower=True)
        self.alpha_ = scipy.linalg.cho_solve(self.L_, y)
        #self.alpha_ = scipy.linalg.solve(K + self.alpha * jnp.eye(X.shape[0]), y, assume_a='pos')

    
    def create_transform_x(self, x):
        self.pipeline_transform_x = []
        if self.normalize:
            scaler = GlobalStandardScaler()
            x = scaler.fit_transform(x)
            self.pipeline_transform_x.append(scaler)
        if self.pca_input_components > 0:
            pca = PCA(n_components=self.pca_input_components)
            x = pca.fit_transform(x)
            self.pipeline_transform_x.append(pca)
        return x
    
    def transform_x(self, x):
        for transform in self.pipeline_transform_x:
            x = transform.transform(x)
        return x
    
    def inverse_transform_x(self, x_transformed):
        for transform in reversed(self.pipeline_transform_x):
            x_transformed = transform.inverse_transform(x_transformed)
        return x_transformed
    
    def create_transform_y(self, y):
        self.pipeline_transform_y = []
        if self.normalize:
            scaler = GlobalStandardScaler()
            y = scaler.fit_transform(y)
            self.pipeline_transform_y.append(scaler)
        if self.pca_output_components > 0:
            pca = PCA(n_components=self.pca_output_components)
            y = pca.fit_transform(y)
            self.pipeline_transform_y.append(pca)
        return y
    
    def transform_y(self, y):
        for transform in self.pipeline_transform_y:
            y = transform.transform(y)
        return y
    def inverse_transform_y(self, y_transformed):
        for transform in reversed(self.pipeline_transform_y):
            y_transformed = transform.inverse_transform(y_transformed)
        return y_transformed


    def predict(self, x, batch_size=None):
        #print("Predicting with shape", x.shape)
        batch_size = batch_size if batch_size is not None else self.batch_size
        #print(batch_size)
        if self.batch_size is not None:
            # If batch size is specified, we will predict in batches
            preds = []
            for i in range(0, x.shape[0], self.batch_size):
                batch_x = x[i:i+self.batch_size]
                batch_x = self.transform_x(batch_x)
                pred = self.kernel(batch_x, self.X, self.parameters)
                pred = jnp.dot(pred, self.alpha_)
                pred = self.inverse_transform_y(pred)
                preds.append(pred)
            return jnp.concatenate(preds, axis=0)
        
        #jax.debug.print("X.shape={x}, X.dtype{dt}", x=x.shape, dt=x.dtype)
        x = self.transform_x(x)
        #print(x.shape)
        #jax.debug.print("X.shape={x}, X.dtype{dt}", x=x.shape, dt=x.dtype)

        K_pred = self.kernel(x, self.X, self.parameters)
        #jax.debug.print("K.shape={k}, K.dtype={dt}", k=K_pred.shape, dt=K_pred.dtype)


        pred = jnp.dot(K_pred, self.alpha_)
        #jax.debug.print("Pred.shape={p}, Pred.dtype={dt}", p=pred.shape, dt=pred.dtype)
        pred = self.inverse_transform_y(pred)

        #jax.debug.print("Pred.shape={p}, Pred.dtype={dt}", p=pred.shape, dt=pred.dtype)
  
        return pred
    
    def score(self, X, y, batch_size=None):
        # The score is the relative L2 loss
        pred = self.predict(X)
        return relative_l2_loss(y, pred)
    
#%%

def cross_val_score(model, X, y, cv=5, scoring = None):
    """
    Perform cross-validation on the model.
    
    Parameters:
    model: The model to evaluate.
    X: Input data.
    y: Target data.
    cv: Number of folds for cross-validation.
    scoring: Scoring function, if None, uses model.score.
    
    Returns:
    Array of scores for each fold.
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=cv)
    scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        if scoring is None:
            score = model.score(X_test, y_test)
        else:
            score = scoring(model, X_test, y_test)
        
        scores.append(score)
    
    return jnp.array(scores)

# %%
