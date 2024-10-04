import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class MultivariateGaussianMixture:
    def __init__(self, num_components=3, num_samples=500, alpha=None, mu_0=None, sigma_0=None, sigma_identity=None, dim=2):
        """
        Initialize the Gaussian Mixture Model generator.
        
        Parameters:
        - num_components: Number of Gaussian components (clusters) in the mixture.
        - num_samples: Number of data samples to generate.
        - alpha: Dirichlet distribution parameters for the mixture proportions.
        - mu_0: Mean of the prior Gaussian for component means (in d-dimensional space).
        - sigma_0: Covariance matrix for the prior Gaussian for component means.
        - sigma_identity: Covariance matrix for the data-generating Gaussian distributions.
        - dim: Dimensionality of the data (e.g., 2 for 2D).
        """
        self.K = num_components
        self.N = num_samples
        self.alpha = alpha if alpha is not None else np.ones(num_components)
        self.mu_0 = mu_0 if mu_0 is not None else np.zeros(dim)
        self.sigma_0 = sigma_0 if sigma_0 is not None else np.eye(dim) * 15
        self.sigma_identity = sigma_identity if sigma_identity is not None else np.eye(dim)
        self.dim = dim

    def generate_data(self):
        """
        Generates data from a Gaussian mixture model with Dirichlet-distributed mixture proportions.
        
        Returns:
        - x: Generated data points of shape (N, d), where N is the number of samples and d is the dimensionality.
        - z: Component assignments for each data point (integers from 0 to K-1).
        - mu: Means of the Gaussian components in the mixture.
        - theta: Mixture proportions generated from the Dirichlet distribution.
        """
        # Step 1: Generate mixture proportions from the Dirichlet distribution
        theta = np.random.dirichlet(self.alpha)
        
        # Step 2: Generate the means of the Gaussian components in d dimensions
        mu = np.random.multivariate_normal(self.mu_0, self.sigma_0, self.K)
        
        # Step 3: Generate the data points
        x = np.zeros((self.N, self.dim))
        z = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            # a) Sample the component index z_i from the Categorical(θ)
            z_i = np.random.choice(self.K, p=theta)
            z[i] = z_i
            # b) Get the mean of the selected component
            mu_zi = mu[z_i]
            # c) Sample the data point x_i from N(μ_{z_i}, σ_identity)
            x[i] = np.random.multivariate_normal(mu_zi, self.sigma_identity)
        
        return x, z, mu, theta

    def plot_data(self, x, z, mu):
        """
        Visualizes the generated data and the Gaussian component means.
        
        Parameters:
        - x: Generated data points.
        - z: Component assignments for each data point.
        - mu: Means of the Gaussian components.
        """
        plt.scatter(x[:, 0], x[:, 1], c=z, cmap='viridis', alpha=0.5)
        plt.scatter(mu[:, 0], mu[:, 1], color='black', marker='*', label='Component Means')
        plt.title(f'Generated Multivariate Gaussian Mixture (dim={self.dim})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.show()

class EMGaussianMixture:
    def __init__(self, num_components=3, tol=1e-6, max_iter=1000):
        """
        Initialize the EM algorithm for Gaussian Mixture Models.
        
        Parameters:
        - num_components: Number of Gaussian components (clusters) in the mixture.
        - tol: Convergence tolerance for the log-likelihood.
        - max_iter: Maximum number of iterations for the algorithm.
        """
        self.K = num_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        """
        Fit the Gaussian Mixture Model using the EM algorithm.
        
        Parameters:
        - X: Data points (n x D), where n is the number of samples and D is the dimensionality.
        
        Returns:
        - pi: Mixing coefficients for each component (1 x K).
        - mu: Means of the Gaussian components (K x D).
        - Sigma: Covariance matrices of the Gaussian components (K x D x D).
        """
        n, D = X.shape
        
        # Initialize parameters randomly
        np.random.seed(42)
        pi = np.random.dirichlet([1] * self.K)  # Mixing coefficients
        mu = np.random.randn(self.K, D)  # Random means for each component
        Sigma = np.array([np.eye(D) for _ in range(self.K)])  # Identity covariances
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Calculate responsibilities
            gamma = self.e_step(X, pi, mu, Sigma)
            
            # M-step: Update parameters
            pi, mu, Sigma = self.m_step(X, gamma)
            
            # Compute log-likelihood
            log_likelihood = self.log_likelihood(X, pi, mu, Sigma)
            
            # Check for convergence
            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"Converged at iteration {iteration}")
                break
            log_likelihood_old = log_likelihood
            
        return pi, mu, Sigma

    def e_step(self, X, pi, mu, Sigma):
        """
        E-step: Compute responsibilities.
        """
        n, D = X.shape
        K = len(pi)
        gamma = np.zeros((n, K))

        for k in range(K):
            mvn_k = multivariate_normal(mean=mu[k], cov=Sigma[k])
            gamma[:, k] = pi[k] * mvn_k.pdf(X)
        
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def m_step(self, X, gamma):
        """
        M-step: Update parameters.
        """
        n, D = X.shape
        K = gamma.shape[1]

        N = np.sum(gamma, axis=0)
        pi = N / n
        mu = np.zeros((K, D))
        Sigma = np.zeros((K, D, D))

        for k in range(K):
            mu[k] = np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / N[k]
            X_centered = X - mu[k]
            Sigma[k] = (X_centered.T @ (gamma[:, k][:, np.newaxis] * X_centered)) / N[k]

        return pi, mu, Sigma

    def log_likelihood(self, X, pi, mu, Sigma):
        """
        Evaluate the log-likelihood of the data under the current parameters.
        """
        n, D = X.shape
        K = len(pi)
        log_likelihood = 0

        for i in range(n):
            temp_sum = 0
            for k in range(K):
                mvn_k = multivariate_normal(mean=mu[k], cov=Sigma[k])
                temp_sum += pi[k] * mvn_k.pdf(X[i])
            log_likelihood += np.log(temp_sum)
        
        return log_likelihood
