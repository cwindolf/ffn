'''Library for whitening and coloring data

Forked from joelouismarino/whiten.py.
Also based on R's rdrr.io/cran/whitening/src/R/whiteningMatrix.R,
and an SAS blog post:
blogs.sas.com/content/iml/2012/02/08/use-the-cholesky-
transformation-to-correlate-and-uncorrelate-variables.html
'''
import numpy as np
import scipy.linalg as la


def whitening_matrix(X, assume_centered=False, method='zca', fudge=1e-8):
    '''Whitening matrix for a dataset `X`

    Returns a matrix `W` such that `(W @ X.T).T` or equivalently `X @ W.T`
    is white, assuming that `X` is centered.

    Arguments
    ---------
    X : np.array
        Data array with shape (n_samples, n_feature1, ..., n_featurek)
    assume_centered : boolean
        If false, center the data.
    method : string
        One of zca, pca, cholesky, zca_cor, pca_cor
    fudge : float
        Small factor to de-emphasize small eigenvalues.

    Returns
    -------
    W : np.array
        n_features x n_features matrix, where
        n_features = n_feature1 * ... * n_featurek
    '''
    # Make sure data is n_samples x n_features
    X = X.reshape((-1, np.prod(X.shape[1:])))

    # Center
    X_centered = X
    if not assume_centered:
        X_centered = X - np.mean(X, axis=0)

    cov = X_centered.T @ X_centered / X_centered.shape[0]

    if method in ['zca', 'pca', 'cholesky']:
        U, sigma, _ = la.svd(cov)
        U = U @ np.diag(np.sign(np.diag(U)))  # Fix sign ambiguity
        invsqrt_sigma = np.diag(1.0 / np.sqrt(sigma + fudge))
        if method == 'zca':
            W = U @ invsqrt_sigma @ U.T
        elif method == 'pca':
            W = invsqrt_sigma @ U.T
        elif method == 'cholesky':
            W = la.cholesky(U @ np.diag(1.0 / sigma) @ U.T, lower=True)
    elif method in ['zca_cor', 'pca_cor']:
        stds = np.sqrt(np.diag(cov))
        corr = cov / np.outer(stds, stds)
        G, theta, _ = la.svd(corr)
        G = G @ np.diag(np.sign(np.diag(G)))  # Fix sign ambiguity
        invsqrt_theta = np.diag(1.0 / np.sqrt(theta + fudge))
        if method == 'zca_cor':
            W = G @ invsqrt_theta @ G.T @ np.diag(1 / stds)
        elif method == 'pca_cor':
            W = invsqrt_theta @ G.T @ np.diag(1 / stds)
    else:
        raise ValueError(f'Whitening method {method} not found.')

    return W


def whiten(X, assume_centered=False, method='zca', fudge=1e-8):
    '''Whiten a dataset `X`

    Arguments
    ---------
    X : np.array
        Dataset with shape (n_samples, n_feature1, ..., n_featurek)
    assume_centered : boolean
        If false, center the data.
    method : string
        One of zca, pca, cholesky, zca_cor, pca_cor
    fudge : float
        Small factor to de-emphasize small eigenvalues.

    Returns
    -------
    Z : an array with the same shape as `X`.
    '''
    # Center
    X_centered = X
    if not assume_centered:
        X_centered = X - np.mean(X, axis=0)

    W = whitening_matrix(
        X_centered, assume_centered=True, method=method, fudge=fudge
    )
    Z = X_centered @ W.T
    Z = Z.reshape(X.shape)
    return Z


def coloring_matrix(X, assume_centered=False, method='cholesky'):
    '''Coloring matrix for a dataset `X`

    Returns a matrix `C` such that `(C @ Z.T).T` or equivalently
    `Z @ C.T` has the correlation structure of `X`, assuming that `Z`
    was white to start with.

    Arguments
    ---------
    X : np.array
        Data array with shape (n_samples, n_feature1, ..., n_featurek)
    assume_centered : boolean
        If false, center the data.
    method : string
        One of zca, pca, cholesky

    Returns
    -------
    C : np.array
        n_features x n_features matrix, where
        n_features = n_feature1 * ... * n_featurek
    '''
    # Make sure data is n_samples x n_features
    X = X.reshape((-1, np.prod(X.shape[1:])))

    # Center
    X_centered = X
    if not assume_centered:
        X_centered = X - np.mean(X, axis=0)

    cov = X_centered.T @ X_centered / X_centered.shape[0]

    if method == 'cholesky':
        C = la.cholesky(cov, lower=True)
    elif method in ['zca', 'pca']:
        U, sigma, _ = la.svd(cov)
        U = U @ np.diag(np.sign(np.diag(U)))  # Fix sign ambiguity
        sqrt_sigma = np.diag(np.sqrt(sigma))
        if method == 'zca':
            C = U @ sqrt_sigma @ U.T
        elif method == 'pca':
            C = U @ sqrt_sigma
    else:
        raise ValueError(f'Coloring method {method} not found.')

    return C
