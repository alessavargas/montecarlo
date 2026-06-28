"""Monte Carlo simulation methods"""

import numpy as np
from abc import ABC, abstractmethod
from numpy.linalg import cholesky, eigh


class MonteCarlo(ABC):
    """Base class for Monte Carlo simulators"""
    
    @abstractmethod
    def simulate(self, returns_df, num_simulations):
        """Simulate returns using the specific method"""
        pass
    
    def _get_mean_and_cov(self, returns_df):
        """Extract mean and covariance"""
        mu = returns_df.mean().values
        sigma = returns_df.cov().values
        return mu, sigma


class CholeskyNormal(MonteCarlo):
    """Cholesky decomposition with Normal distribution"""
    
    def simulate(self, returns_df, num_simulations):
        mu, sigma = self._get_mean_and_cov(returns_df)
        
        # Cholesky decomposition
        L = cholesky(sigma + 1e-15 * np.eye(sigma.shape[0]))
        
        # Generate normal random numbers
        Z = np.random.normal(0, 1, (num_simulations, len(returns_df.columns)))
        
        # Correlate using Cholesky
        simulated = mu + Z @ L.T
        return simulated


class CholeskyEmpirical(MonteCarlo):
    """Cholesky decomposition with Empirical (Bootstrap) distribution"""
    
    def simulate(self, returns_df, num_simulations):
        mu, sigma = self._get_mean_and_cov(returns_df)
        
        # Bootstrap historical returns
        indices = np.random.choice(len(returns_df), size=num_simulations, replace=True)
        historical_returns = returns_df.iloc[indices].values
        
        # Cholesky decomposition
        L = cholesky(sigma + 1e-15 * np.eye(sigma.shape[0]))
        
        # Small Gaussian perturbation for smoothing
        Z = np.random.normal(0, 1, (num_simulations, len(returns_df.columns))) @ (0.1 * L.T)
        
        # Center and add perturbation
        simulated = mu + (historical_returns - mu) + Z
        return simulated


class PCANormal(MonteCarlo):
    """Principal Component Analysis with Normal distribution"""
    
    def simulate(self, returns_df, num_simulations, var_expl=0.999):
        mu, sigma = self._get_mean_and_cov(returns_df)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(sigma)
        
        # Sort in descending order
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Select components explaining var_expl variance
        cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        k = np.searchsorted(cumsum, var_expl) + 1
        
        # Principal components matrix
        A = eigenvectors[:, :k] @ np.diag(np.sqrt(np.maximum(eigenvalues[:k], 0)))
        
        # Generate random factors
        Z = np.random.normal(0, 1, (num_simulations, k))
        
        # Reconstruct returns
        simulated = mu + Z @ A.T
        return simulated


class PCAEmpirical(MonteCarlo):
    """Principal Component Analysis with Empirical (Bootstrap) distribution"""
    
    def simulate(self, returns_df, num_simulations, var_expl=0.999):
        mu, sigma = self._get_mean_and_cov(returns_df)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(sigma)
        
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        k = np.searchsorted(cumsum, var_expl) + 1
        
        A = eigenvectors[:, :k] @ np.diag(np.sqrt(np.maximum(eigenvalues[:k], 0)))
        A_inv = np.linalg.pinv(A)
        
        # Project historical data to principal components
        Z_hist = returns_df.values @ A_inv.T
        
        # Bootstrap components
        indices = np.random.choice(Z_hist.shape[0], size=num_simulations, replace=True)
        Z = Z_hist[indices, :] + 0.1 * np.random.normal(0, 1, (num_simulations, k))
        
        # Reconstruct
        simulated = mu + Z @ A.T
        return simulated
