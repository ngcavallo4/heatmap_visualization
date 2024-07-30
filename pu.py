import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Create some example data
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) ** 10 + np.cos(10 + Y * X) * np.cos(X)

# Scatter plot data
scatter_x = np.random.uniform(-3, 3, 50)
scatter_y = np.random.uniform(-3, 3, 50)
scatter_z = np.sin(scatter_x) ** 10 + np.cos(10 + scatter_y * scatter_x) * np.cos(scatter_x)

# Create the contour plot
fig, ax = plt.subplots()

# Use contourf to create filled contours
cmap = plt.get_cmap('viridis')
contourf_set = ax.contourf(X, Y, Z, levels=np.linspace(-1, 1, 10), cmap=cmap)

# Add contour lines with the same colormap
contour_set = ax.contour(X, Y, Z, levels=contourf_set.levels, colors='k')

# Optionally, add labels to contour lines
ax.clabel(contour_set, inline=True, fontsize=8)

# Add a scatter plot on top
scatter = ax.scatter(scatter_x, scatter_y, c=scatter_z, cmap=cmap, edgecolor='k')

# Add a colorbar for the filled contours
cbar = fig.colorbar(contourf_set, ax=ax)
cbar.set_label('Value')

# Customize plot
ax.set_title('Contour Plot with Scatter Plot on Top')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

# Show plot
plt.show()