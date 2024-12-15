import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

class RandomSmoothLandscape2D:
    def __init__(self, m, sigma=5, seed=None):
        """
        m: Number of states per gene (for each of the two genes)
        sigma: Standard deviation for Gaussian kernel (higher values mean smoother)
        seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Number of states for each of the two genes
        self.m = m
        self.sigma = sigma  # Gaussian smoothing parameter
        
        # Generate random fitness values for each state of gene combinations
        self.fitness_table = self._generate_random_fitness_table()
        
        # Apply Gaussian smoothing to the fitness values
        self.smooth_fitness_table()

    def _generate_random_fitness_table(self):
        """Generate random fitness values for each possible combination of gene states."""
        fitness_table = np.random.uniform(0, 1, (self.m, self.m))
        return fitness_table

    def smooth_fitness_table(self):
        """Apply Gaussian smoothing to the fitness table to achieve a smoother landscape."""
        self.fitness_table = gaussian_filter(self.fitness_table, sigma=self.sigma)

    def evaluate(self, gene_values):
        """
        Get the fitness of a specific combination of two gene values.
        
        gene_values: A tuple of two gene values (e.g., (gene1_value, gene2_value)).
        
        Returns the fitness score of the given gene configuration.
        """
        i, j = gene_values
        return self.fitness_table[i, j]
    
    def generate_landscape(self):
        """Return the smoothed 2D array representing the fitness landscape."""
        return self.fitness_table

# Parameters
m = 100  # Number of states per gene for a high-resolution landscape
sigma = 10  # Gaussian smoothing factor for a smoother landscape

# Create the smoothed random landscape with Gaussian smoothing and visualize it
landscape = RandomSmoothLandscape2D(m, sigma=sigma, seed=42)
fitness_landscape = landscape.generate_landscape()

# Plotting the 3D Smoothed Random Landscape
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create grid for x and y coordinates
x = np.arange(m)
y = np.arange(m)
x, y = np.meshgrid(x, y)

# Plot a surface with the fitness values as the height (z-axis)
ax.plot_surface(x, y, fitness_landscape, cmap='viridis', edgecolor='k', alpha=0.8)

# Labels and title
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_zlabel("Utility")
ax.set_title(f"3D Gaussian-Smooth Random Landscape with Sigma={sigma}")

plt.show()
