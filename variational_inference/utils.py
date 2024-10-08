import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os


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
        self.sigma_0 = sigma_0 if sigma_0 is not None else np.eye(dim) * 100
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
    def __init__(self, num_components=3, tol=1e-6, max_iter=100):
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
        self.pi = None
        self.mu = None
        self.Sigma = None

    def fit(self, X, save_path='em_gif', threshold=0.5):
        """
        Fit the Gaussian Mixture Model using the EM algorithm and save images at each iteration for GIF creation.
        
        Parameters:
        - X: Data points (n x D), where n is the number of samples and D is the dimensionality.
        - save_path: Path to save the generated images for the GIF.
        - threshold: Classification threshold based on responsibilities.
        
        Returns:
        - pi: Mixing coefficients for each component (1 x K).
        - mu: Means of the Gaussian components (K x D).
        - Sigma: Covariance matrices of the Gaussian components (K x D x D).
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        n, D = X.shape
        
        # Initialize parameters randomly
        np.random.seed(42)
        self.pi = np.random.dirichlet([1] * self.K)  # Mixing coefficients
        self.mu = np.random.randn(self.K, D)  # Random means for each component
        self.Sigma = np.array([np.eye(D) for _ in range(self.K)])  # Identity covariances
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Calculate responsibilities
            gamma = self.e_step(X, self.pi, self.mu, self.Sigma)
            
            # Classify based on the threshold
            classifications = np.argmax(gamma, axis=1)

            # M-step: Update parameters
            self.pi, self.mu, self.Sigma = self.m_step(X, gamma)
            
            # Plot and save the current state
            self.plot_iteration(X, classifications, self.mu, iteration, save_path)
            
            # Compute log-likelihood
            log_likelihood = self.log_likelihood(X, self.pi, self.mu, self.Sigma)
            
            # Check for convergence
            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"Converged at iteration {iteration}")
                break
            log_likelihood_old = log_likelihood
            
        # Generate GIF after fitting
        self.generate_gif(save_path)

        return self.pi, self.mu, self.Sigma

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

    def predict(self, x):
        """
        Predict the component (class) for a single data point x.
        
        Parameters:
        - x: Single data point of shape (D,), where D is the dimensionality.
        
        Returns:
        - class_label: The predicted component (class) for the given data point.
        """
        responsibilities = np.zeros(self.K)
        for k in range(self.K):
            mvn_k = multivariate_normal(mean=self.mu[k], cov=self.Sigma[k])
            responsibilities[k] = self.pi[k] * mvn_k.pdf(x)
        
        responsibilities /= np.sum(responsibilities)  # Normalize responsibilities
        
        # Return the class with the highest responsibility
        class_label = np.argmax(responsibilities)
        return class_label

    def plot_iteration(self, X, classifications, mu, iteration, save_path):
        """
        Plot and save the classification of the data points at each iteration.
        
        Parameters:
        - X: Data points.
        - classifications: Component assignments for each data point.
        - mu: Current means of the Gaussian components.
        - iteration: Current iteration number.
        - save_path: Path to save the generated image.
        """
        plt.scatter(X[:, 0], X[:, 1], c=classifications, cmap='viridis', alpha=0.5)
        plt.scatter(mu[:, 0], mu[:, 1], color='black', marker='*', label='Component Means')
        plt.title(f'Iteration {iteration}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig(f'{save_path}/iter_{iteration}.png')
        plt.close()

    def generate_gif(self, save_path):
        """
        Generate a GIF from saved images.
        
        Parameters:
        - save_path: Path where images are stored.
        """
        images = []
        filenames = sorted([f for f in os.listdir(save_path) if f.endswith('.png')])
        for filename in filenames:
            images.append(imageio.imread(os.path.join(save_path, filename)))
        imageio.mimsave(f'{save_path}/em_iterations.gif', images, duration=0.5)

class ExponentialNormalNDGenerator:
    def __init__(self, lam, dim=2):
        """
        Initialize the generator with the given lambda for the exponential distribution.
        
        Parameters:
        - lam: The lambda parameter for the exponential distribution.
        - dim: The number of dimensions for the data (default: 2).
        """
        self.lam = lam
        self.dim = dim

    def generate_data(self, num_samples=100):
        """
        Generate data following the process:
        1. z ~ Exponential(lam)
        2. x ~ N(mu=z, sigma^2=1) (for each dimension)
        
        Parameters:
        - num_samples: The number of data points to generate.
        
        Returns:
        - z: Values generated from the exponential distribution.
        - x: Values generated from the normal distribution in `dim` dimensions.
        """
        # Step 1: Generate z from Exponential(lam) for each sample
        z = np.random.exponential(scale=1/self.lam, size=num_samples)

        # Step 2: Generate x for each dimension, with N(mu=z, sigma^2=1)
        x = np.zeros((num_samples, self.dim))
        for d in range(self.dim):
            x[:, d] = np.random.normal(loc=z, scale=1, size=num_samples)
        
        return z, x

    def plot_data(self, z, x):
        """
        Plot the generated data. For dim=2, it creates a contour plot.
        
        Parameters:
        - z: The values generated from the exponential distribution.
        - x: The values generated from the normal distribution.
        """
        if self.dim == 1:
            # Plot for 1D case
            plt.figure(figsize=(10, 5))

            # Plot z
            plt.subplot(1, 2, 1)
            plt.hist(z, bins=30, color='blue', alpha=0.7)
            plt.title('Exponential Distribution (z values)')
            plt.xlabel('z')
            plt.ylabel('Frequency')

            # Plot x
            plt.subplot(1, 2, 2)
            plt.hist(x[:, 0], bins=30, color='green', alpha=0.7)
            plt.title('Normal Distribution (x values, dim=1)')
            plt.xlabel('x')
            plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()

        elif self.dim == 2:
            # Create a 2D contour plot
            plt.figure(figsize=(8, 6))

            # Create 2D histogram (density) for contour plot
            hist, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], bins=30, density=True)

            # Generate the grid for contour plotting
            X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

            # Create contour plot
            plt.contourf(X, Y, hist.T, levels=20, cmap='viridis')
            plt.colorbar(label='Density')

            plt.title('Contour Plot of x (dim=2)')
            plt.xlabel('$x_0$')
            plt.ylabel('$x_1$')
            plt.grid(True)
            plt.show()

        else:
            raise ValueError("Plotting is only supported for dim=1 or dim=2")

    def plot_scatter(self, z, x):
        """
        Scatter plot of z vs x for dim=1 or dim=2 to show the relationship.
        
        Parameters:
        - z: The values generated from the exponential distribution.
        - x: The values generated from the normal distribution.
        """
        if self.dim == 1:
            # Plot scatter for 1D
            plt.figure(figsize=(6, 6))
            plt.scatter(z, x[:, 0], alpha=0.5, color='purple')
            plt.title('Scatter Plot of z vs x (dim=1)')
            plt.xlabel('z (Exponential)')
            plt.ylabel('x (Normal)')
            plt.grid(True)
            plt.show()

        elif self.dim == 2:
            # Plot scatter for 2D
            plt.figure(figsize=(6, 6))
            plt.scatter(z, x[:, 0], alpha=0.5, color='blue', label='Dimension 1')
            plt.scatter(z, x[:, 1], alpha=0.5, color='green', label='Dimension 2')
            plt.title('Scatter Plot of z vs x (dim=2)')
            plt.xlabel('z (Exponential)')
            plt.ylabel('x (Normal)')
            plt.legend()
            plt.grid(True)
            plt.show()

        else:
            raise ValueError("Scatter plotting is only supported for dim=1 or dim=2")
