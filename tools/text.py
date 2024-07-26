if field_name == "Interpolation":
    im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]), alpha=alpha)
else:
    im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))

# if field_name == "Interpolation":
im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]), alpha=alpha)
# else:
#     im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
ax.set_xlim([x_range[0], x_range[1]])
ax.set_ylim([y_range[0], y_range[1]])
if match_scale:
    im.norm.autoscale([colormin, colormax])
if field_name == "Interpolation":
    ax.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis', s=15)
ax.set_title(f'{field_name} – {title}')
ax.ticklabel_format(useOffset=False)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar = self.fig.colorbar(im, ax=ax, shrink=0.7)
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

# Set x-axis label
offset_text = ax.xaxis.get_offset_text()
offset_text.set_size(7)
offset_text = ax.yaxis.get_offset_text()
offset_text.set_size(7)
ax.set_xlabel('Latitude(º)',fontsize=10)
ax.xaxis.set_label_coords(0.5, -0.16)
ax.set_ylabel("Longitude(º)",loc='center',fontsize=10)