import numpy as np
from scipy.interpolate import Rbf
import multiprocessing as mp

# Define the Gaussian interpolation function
def gaussian_interpolation(x, y, z, xi, yi):
    rbf = Rbf(x, y, z, function='gaussian')
    zi = rbf(xi, yi)
    return zi

# Function to call in parallel
def parallel_task(args):
    partition_number, x, y, z, xi, yi = args
    # Perform Gaussian interpolation on the dataset
    result = gaussian_interpolation(x, y, z, xi, yi)
    return result

if __name__ == '__main__':
    # Example data
    x = np.random.rand(300)
    y = np.random.rand(300)
    z = np.random.rand(300)
    xi = np.random.rand(200)
    yi = np.random.rand(200)

    # Number of parallel tasks
    num_tasks = 3

    # Create a list of tasks to run in parallel
    tasks = [(i, x, y, z, xi, yi) for i in range(num_tasks)]

    # Create a multiprocessing Pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Map the tasks to the pool and run them in parallel
        results = pool.map(parallel_task, tasks)

    # Combine or process the results as needed
    # For example, you can concatenate the results if they are arrays
    combined_results = np.vstack(results)

    print(combined_results)
