import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import json
import os


class Landscape:
    def __init__(self, size=41, sigma=5, seed=None, method="boostGlobal"):
        
        self.size = size
        self.sigma = sigma  # Gaussian smoothing parameter
        self.values = self._generate_landscape(seed=seed, method=method)
        
    def _generate_random_landscape_dampenLocal(self):
        
        #kernal_size = 2 * np.ceil(3 * self.sigma) + 1

        # Generate uniform random values for landscape
        random_landscape = np.random.uniform(0, 1, (self.size, self.size))

        # Apply Gaussian smoothing to the landscape
        smoothed_landscape = gaussian_filter(random_landscape, sigma=self.sigma, mode='wrap')

        # Normalize Landscape to values between 0 and 1
        normalized_landscape = (smoothed_landscape - np.min(smoothed_landscape)) / (np.max(smoothed_landscape) - np.min(smoothed_landscape))

        # Multiply all hills that are not the global maximum by 0.9
        normalized_dampened_landscape = _dampen_local_maxima(normalized_landscape)

        return normalized_dampened_landscape

    def _generate_random_landscape_boostGlobal(self):

        #kernal_size = 2 * np.ceil(3 * self.sigma) + 1

        # Generate uniform random values for landscape
        random_landscape = np.random.uniform(0, 1, (self.size, self.size))

        # Apply Gaussian smoothing to the landscape
        smoothed_landscape = gaussian_filter(random_landscape, sigma=self.sigma, mode='wrap')

        # Find global maximum
        max_index = np.unravel_index(np.argmax(smoothed_landscape, axis=None), smoothed_landscape.shape)
        # print(f"Global Maximum at: {max_index}")
        # print(f"Global Maximum Value: {normalized_landscape[max_index]}")

        # Set global maximum to 1
        random_landscape[max_index] = 2

        # Apply Gaussian smoothing to the landscape
        smoothed_landscape = gaussian_filter(random_landscape, sigma=self.sigma, mode='wrap')

        # Normalize Landscape to values between 0 and 0.9
        normalized_landscape = (smoothed_landscape - np.min(smoothed_landscape)) / (np.max(smoothed_landscape) - np.min(smoothed_landscape))


        return normalized_landscape

    def _generate_onePeak_landscape(self):

        # Generate flat landscape
        landscape = np.full((self.size, self.size), 0.1, dtype=float)

        # Set one random peak to 1
        peak = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        landscape[peak] = 1

        # Apply Gaussian smoothing to the landscape
        smoothed_landscape = gaussian_filter(landscape, sigma=self.sigma, mode='wrap')

        # Normalize Landscape to values between 0.1 and 1
        normalized_landscape = (smoothed_landscape - np.min(smoothed_landscape)) / (np.max(smoothed_landscape) - np.min(smoothed_landscape))

        # Clamp values between 0.1 and 1
        normalized_landscape = np.clip(normalized_landscape, 0.1, 1)
        
        return normalized_landscape

    def _generate_landscape(self, seed=None, method="boostGlobal"):

        if seed is not None:
            np.random.seed(seed)


        if method == "dampenLocal":
            return self._generate_random_landscape_dampenLocal()
        elif method == "boostGlobal":
            return self._generate_random_landscape_boostGlobal()
        elif method == "onePeak":
            return self._generate_onePeak_landscape()
        else:
            raise ValueError("Invalid method. Choose 'dampenLocal', 'boostGlobal', or 'onePeak'.")

    def __getitem__(self, key):
        row, col = key
        row = row % self.values.shape[0]  # Wrap row index
        col = col % self.values.shape[1]  # Wrap column index
        return self.values[row, col]

    def __setitem__(self, key, value):
        row, col = key
        row = row % self.values.shape[0]  # Wrap row index
        col = col % self.values.shape[1]  # Wrap column index
        self.values[row, col] = value

    def __repr__(self):
        return repr(self.values)
        
def find_local_maxima(landscape):
    # Find local maxima
    local_maxima = []
    for i in range(0, landscape.size):
        for j in range(0, landscape.size):
            if landscape[i, j] > landscape[i - 1, j] and landscape[i, j] > landscape[i + 1, j] and landscape[i, j] > landscape[i, j - 1] and landscape[i, j] > landscape[i, j + 1]:
                if landscape[i, j] > landscape[i - 1, j - 1] and landscape[i, j] > landscape[i - 1, j + 1] and landscape[i, j] > landscape[i + 1, j - 1] and landscape[i, j] > landscape[i + 1, j + 1]:
                    local_maxima.append(((i, j), landscape[(i, j)]))
    
    # Print values of local maxima
    #for maxima, _ in local_maxima:
    #    print(f"Local maxima at {maxima}: {landscape[maxima]}")
    
    return local_maxima

def find_local_maxima_3d(landscape):
    # Find local maxima in 3D landscape
    local_maxima = []
    for i in range(0, landscape.size):
        for j in range(0, landscape.size):
            for k in range(0, landscape.size):
                if (landscape[i, j, k] > landscape[i - 1, j, k] and
                    landscape[i, j, k] > landscape[i + 1, j, k] and
                    landscape[i, j, k] > landscape[i, j - 1, k] and
                    landscape[i, j, k] > landscape[i, j + 1, k] and
                    landscape[i, j, k] > landscape[i, j, k - 1] and
                    landscape[i, j, k] > landscape[i, j, k + 1] and

                    landscape[i, j, k] > landscape[i - 1, j - 1, k] and
                    landscape[i, j, k] > landscape[i - 1, j + 1, k] and
                    landscape[i, j, k] > landscape[i + 1, j - 1, k] and
                    landscape[i, j, k] > landscape[i + 1, j + 1, k] and

                    landscape[i, j, k] > landscape[i - 1, j - 1, k - 1] and
                    landscape[i, j, k] > landscape[i - 1, j + 1, k - 1] and
                    landscape[i, j, k] > landscape[i + 1, j - 1, k - 1] and
                    landscape[i, j, k] > landscape[i + 1, j + 1, k - 1] and

                    landscape[i, j, k] > landscape[i - 1, j - 1, k + 1] and
                    landscape[i, j, k] > landscape[i - 1, j + 1, k + 1] and
                    landscape[i, j, k] > landscape[i + 1, j - 1, k + 1] and
                    landscape[i, j, k] > landscape[i + 1, j + 1, k + 1] and

                    landscape[i, j, k] > landscape[i, j - 1, k - 1] and
                    landscape[i, j, k] > landscape[i, j + 1, k - 1] and
                    landscape[i, j, k] > landscape[i, j - 1, k + 1] and
                    landscape[i, j, k] > landscape[i, j + 1, k + 1] and

                    landscape[i, j, k] > landscape[i - 1, j, k + 1] and
                    landscape[i, j, k] > landscape[i - 1, j, k - 1] and
                    landscape[i, j, k] > landscape[i + 1, j, k + 1] and
                    landscape[i, j, k] > landscape[i + 1, j, k - 1]
                    ):
                    local_maxima.append(((i, j, k), landscape[(i, j, k)]))
    
    return local_maxima

def _dampen_local_maxima(landscape):
    max_index = np.unravel_index(np.argmax(landscape, axis=None), landscape.shape)

    explored = set()
    queued = [max_index]

    # BFS to identify the global hill
    while queued:
        current = queued.pop(0)
        if current in explored:
            continue
        explored.add(current)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor_x = (current[0] + i) % landscape.shape[0]
                neighbor_y = (current[1] + j) % landscape.shape[1]
                neighbor = (neighbor_x, neighbor_y)
                if (
                    0 <= neighbor[0] < landscape.shape[0] and
                    0 <= neighbor[1] < landscape.shape[1] and
                    neighbor not in explored and
                    landscape[neighbor] < landscape[current]
                ):
                    queued.append(neighbor)

    # Create a mask for non-hill areas
    mask = np.ones_like(landscape, dtype=bool)
    for point in explored:
        mask[point] = False

    # Apply dampening
    landscape[mask] *= 0.9

    return landscape

class Landscape3D:
    def __init__(self, size=41, sigma=5, seed=None, method="boostGlobal"):
        self.size = size
        self.sigma = sigma  # Gaussian smoothing parameter
        self.values = self._generate_landscape(seed=seed, method=method)
        

    def _generate_random_landscape(self):

        # Generate uniform random values for landscape
        random_landscape = np.random.uniform(0, 1, (self.size, self.size, self.size))

        # Apply Gaussian smoothing to the landscape
        smoothed_landscape = gaussian_filter(random_landscape, sigma=self.sigma, mode='wrap')

        # Find global maximum
        max_index = np.unravel_index(np.argmax(smoothed_landscape, axis=None), smoothed_landscape.shape)
        # print(f"Global Maximum at: {max_index}")
        # print(f"Global Maximum Value: {normalized_landscape[max_index]}")

        # Set global maximum to 1
        random_landscape[max_index] = 3

        # Apply Gaussian smoothing to the landscape
        smoothed_landscape = gaussian_filter(random_landscape, sigma=self.sigma, mode='wrap')

        # Normalize Landscape to values between 0 and 0.9
        normalized_landscape = (smoothed_landscape - np.min(smoothed_landscape)) / (np.max(smoothed_landscape) - np.min(smoothed_landscape))


        return normalized_landscape

    def _generate_onePeak_landscape(self):

        # Generate flat landscape
        landscape = np.full((self.size, self.size, self.size), 0.1, dtype=float)

        # Set one random peak to 1
        peak = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        landscape[peak] = 1

        # Apply Gaussian smoothing to the landscape
        smoothed_landscape = gaussian_filter(landscape, sigma=self.sigma, mode='wrap')

        # Normalize Landscape to values between 0.1 and 1
        normalized_landscape = (smoothed_landscape - np.min(smoothed_landscape)) / (np.max(smoothed_landscape) - np.min(smoothed_landscape))

        # Clamp values between 0.1 and 1
        normalized_landscape = np.clip(normalized_landscape, 0.1, 1)
        
        return normalized_landscape

    def _generate_landscape(self, seed=None, method="boostGlobal"):

        if seed is not None:
            np.random.seed(seed)

        if method == "boostGlobal":
            return self._generate_random_landscape()
        elif method == "onePeak":
            return self._generate_onePeak_landscape()
        else:
            raise ValueError("Invalid method. Choose, 'boostGlobal', or 'onePeak'.")

    def __getitem__(self, key):
        row, col, z = key
        row = row % self.values.shape[0]  # Wrap row index
        col = col % self.values.shape[1]  # Wrap column index
        z = z % self.values.shape[2]  # Wrap z index
        return self.values[row, col, z]

    def __setitem__(self, key, value):
        row, col, z = key
        row = row % self.values.shape[0]  # Wrap row index
        col = col % self.values.shape[1]  # Wrap column index
        z = z % self.values.shape[2]  # Wrap z index
        self.values[row, col, z] = value

    def __repr__(self):
        return repr(self.values)





def export_landscape_to_json(numpy_array, output_path='volume_data.json'):
    """
    Export a 3D NumPy array to JSON for visualization with the React component.
    
    Parameters:
    -----------
    numpy_array : numpy.ndarray
        3D NumPy array with values between 0 and 1
    output_path : str
        Path to save the JSON file
    """
    # Ensure the array is 3D
    if numpy_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {numpy_array.ndim}D array")
    
    # Ensure values are between 0 and 1
    if numpy_array.min() < 0 or numpy_array.max() > 1:
        print("Warning: Values outside [0,1] range. Normalizing data...")
        # Normalize to [0,1] range
        numpy_array = (numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min())
    
    # Prepare data for JSON
    data = {
        'data': numpy_array.tolist(),
        'dimensions': numpy_array.shape
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Data exported to {output_path}")
    print(f"Array shape: {numpy_array.shape}")
    print(f"Value range: [{numpy_array.min():.4f}, {numpy_array.max():.4f}]")



if __name__ == "__main__":

    # Parameters
    size = 15  
    sigma = 3  # Gaussian smoothing factor for a smoother landscape
    method = "bg"  # "dampenLocal" ("dl") or "boostGlobal" ("bg") or "onePeak" ("op")

    dim2d = False
    viz = False  # False for statistics of many landscape, True for visualization of a single landscape
    viz_seed = None


    if method == "dl":
        method = "dampenLocal"
    elif method == "bg":
        method = "boostGlobal"
    elif method == "op":
        method = "onePeak"



    if not dim2d:
        landscape = Landscape3D(size, sigma=sigma, seed=viz_seed)

        # Export the landscape to JSON for visualization
        export_landscape_to_json(landscape.values, output_path=f'grid{size}_sigma{sigma}.json')



    if dim2d:
        if not viz:

            second_peaks = []
            third_peaks = []
            for i in range(100):

                #landscape = Landscape(size, sigma=sigma, seed=i, method=method)
                landscape = Landscape3D(size, sigma=sigma, seed=i)

                # Find local maxima
                local_maxima = find_local_maxima_3d(landscape)

                # Find second highest peak
                local_maxima.sort(key=lambda x: x[1], reverse=True)
                try:
                    peak2 = local_maxima[1]
                    second_peaks.append(peak2[1])

                    if peak2[1] > 0.9:
                        print(f"peak2: {peak2[1]}", i)
                except:
                    pass

                try:
                    peak3 = local_maxima[2]
                    third_peaks.append(peak3[1])
                except:
                    pass


            print(f"average second peak: {np.mean(second_peaks)}")
            print(f"std second peak: {np.std(second_peaks)}")
            print(f"max second peak: {np.max(second_peaks)}")
            print(f"min second peak: {np.min(second_peaks)}")


        else:
            landscape = Landscape(size, sigma=sigma, seed=viz_seed, method=method)

            local_maxima = find_local_maxima(landscape)
            local_maxima.sort(key=lambda x: x[1], reverse=True)

            for i, peak in enumerate(local_maxima):
                print(i, peak)

            # Plotting the 3D Smoothed Random Landscape
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # Create grid for x and y coordinates
            x = np.arange(size)
            y = np.arange(size)
            x, y = np.meshgrid(x, y)

            # Plot a surface with the fitness values as the height (z-axis)
            ax.plot_surface(x, y, landscape.values, cmap='viridis', edgecolor='k', alpha=0.8)

            # Labels and title
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Utility")
            ax.set_title(f"{size}x{size}, Sigma={sigma}, Method={method}")

            plt.show()
