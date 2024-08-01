import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import multiprocessing 
from heatmap_simplified import Heatmap

# Example data
x = np.random.uniform(0, 10, 20)
y = np.random.uniform(0, 10, 20)
stiff = np.random.uniform(0, 10, 20)

# Gaussian Process parameters
length_scale = {"val": 1.0, "bounds": (1e-2, 1e1)}
sigma_f = {"val": 1.0, "bounds": (1e-2, 1e1)}
noise_level = {"val": 1e-1, "bounds": (1e-4, 1e1)}
nu = 1.5

# Create a Heatmap instance
heatmap = Heatmap(length_scale, sigma_f, noise_level, nu, latlon_to_m=False)

# Update the heatmap with the generated data
z_pred = heatmap.update_heatmap(x, y, stiff)

# First figure
fig1, ax1 = plt.subplots()

# Heatmap
im1 = ax1.imshow(heatmap.z_pred, origin='lower', cmap='viridis', 
                 extent=(heatmap.x_range[0], heatmap.x_range[1], heatmap.y_range[0], heatmap.y_range[1]))

# Scatter plot with rescaled colormap
norm1 = mcolors.Normalize(vmin=np.min(stiff), vmax=np.max(stiff))
sc1 = ax1.scatter(heatmap.x, heatmap.y, c=heatmap.stiff, cmap='viridis', edgecolors='k', s=20, norm=norm1)

# Adding colorbars for both plots
cbar_z1 =fig1.colorbar(im1, ax=ax1)
cbar_sc1 =fig1.colorbar(sc1, ax=ax1)
cbar_sc1.set_label('Stiffness', rotation=270, labelpad = 15)
cbar_z1.set_label('Heatmap Intensity', rotation=270, labelpad = 15)

# Second set of example data
x1 = np.random.uniform(0, 10, 20)
y1 = np.random.uniform(0, 10, 20)
stiff1 = np.random.uniform(0, 10, 20)

# Update the heatmap with the second set of generated data
z_pred1 = heatmap.update_heatmap(x1, y1, stiff1)

# Second figure
fig2, ax2 = plt.subplots()

# Heatmap
im2 = ax2.imshow(heatmap.z_pred, origin='lower', cmap='viridis', 
                 extent=(heatmap.x_range[0], heatmap.x_range[1], heatmap.y_range[0], heatmap.y_range[1]))

# Scatter plot with rescaled colormap
norm2 = mcolors.Normalize(vmin=np.min(stiff1), vmax=np.max(stiff1))
sc2 = ax2.scatter(heatmap.x, heatmap.y, c=heatmap.stiff, cmap='viridis', edgecolors='k', s=20, norm=norm2)

# Adding colorbars for both plots
cbar_z2 = fig2.colorbar(im2, ax=ax2, label='Heatmap Intensity')
cbar_sc2 = fig2.colorbar(sc2, ax=ax2)
cbar_sc2.set_label('Stiffness', rotation=270, labelpad = 15)
cbar_z2.set_label('Heatmap Intensity', rotation=270, labelpad = 15)

plt.show()

