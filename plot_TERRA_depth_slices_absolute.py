import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from terratools.terra_model import read_netcdf
import matplotlib.gridspec as gridspec
import os

def load_plate_boundaries(file_path):
    boundaries = []
    current_boundary = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_boundary:
                    boundaries.append(np.array(current_boundary))
                    current_boundary = []
            else:
                parts = line.split()
                lon, lat = float(parts[0]), float(parts[1])
                current_boundary.append([lon, lat])
        if current_boundary:
            boundaries.append(np.array(current_boundary))
    return boundaries

def plot_plate_boundaries(ax, boundaries, color='lime', exclude_segments=[]):
    for i, boundary in enumerate(boundaries):
        if i in exclude_segments:
            continue
        if len(boundary) > 1:
            lons, lats = zip(*boundary)
            ax.plot(lons, lats, color=color, linewidth=1, transform=ccrs.PlateCarree(), zorder=10)

def plot_depth_slice(model_path, field, radius, depth_label, cmap, cbar_label, cbar_height, cbar_pad, cbar_length, plate_boundaries, vmin=None, vmax=None, output_filename="example_depth_slice.png"):
    # Read in the model
    model = read_netcdf([model_path])

    # Plot the layer for the given radius and return the colorbar
    fig, ax, cbar = model.plot_layer(field=field, radius=radius, depth=False, show=False, return_cbar=True)
    fig.set_size_inches(8, 6)
    #ax.set_title(f"{int(depth_label)} km depth, min={data.min()}, max={data.max()}", fontsize=30)
    #ax.set_title(f"{int(depth_label)} km depth", fontsize=30)

    # Change the colormap to the specified cmap and set vmin and vmax if provided
    for img in ax.get_images():
        img.set_cmap(cmap)
        if vmin is not None and vmax is not None:
            img.set_clim(vmin, vmax)
        else:
            # Print data range for debugging
            data = img.get_array()
            print(f"Data range for {field} at {depth_label} km depth: min={data.min()}, max={data.max()}")
            img.set_clim(data.min(), data.max())  # Set vmin and vmax based on data range


        data = ax.get_images()[0].get_array().data
        avg_value = np.mean(data)
        perturbation = (data - avg_value) / avg_value * 100

        if cbar:
            cbar.remove()

    # Plot the layer for the given radius and return the colorbar
    #fig, ax, cbar = model.plot_layer(field=field, radius=radius, depth=False, show=False, return_cbar=True)
    #fig.set_size_inches(8, 6)
    ax.set_title(f"{int(depth_label)} km, min={data.min():.3g}, max={data.max():.3g}", fontsize=30)
    #ax.set_title(f"{int(depth_label)} km depth", fontsize=30)
    min_value = data.min()
    max_value = data.max()
    formatted_min = f"{min_value:.3g}" if isinstance(min_value, float) and not min_value.is_integer() else str(int(min_value))
    formatted_max = f"{max_value:.3g}" if isinstance(max_value, float) and not max_value.is_integer() else str(int(max_value))
    ax.set_title(f"{int(depth_label)} km, min={formatted_min}, max={formatted_max}", fontsize=26)

    # Plot plate boundaries, excluding problematic segments
    plot_plate_boundaries(ax, plate_boundaries['ridge'], color='lime', exclude_segments=[72, 99])
    plot_plate_boundaries(ax, plate_boundaries['trench'], color='lime', exclude_segments=[110])
    plot_plate_boundaries(ax, plate_boundaries['transform'], color='lime')

# Conditional logic based on depth
    if depth == 2799.6876:  # Check if the depth is exactly 2800 km
    # Check if a colorbar already exists and remove it
        if cbar:
            #cbar.remove()

        # Calculate symmetric color limits based on field and depth
            if field == "vs":
                min, max = data.min(), data.max()
            elif field == "vp":
                min, max = data.min(), data.max()
            elif field == "density":
                min, max = data.min(), data.max()
            elif field == "t":
                min, max = data.min(), data.max()

            #ax.set_title(f"{int(depth_label)} km depth, max="(vmax), fontsize=30)
        ax.get_images()[0].set_clim(min, max)
        print('===========',max)
        #ax.set_title(f"{int(depth_label)} km, min={data.min():.3g}, max={data.max():.3g}", fontsize=30)
        min_value = data.min()
        max_value = data.max()
        formatted_min = f"{min_value:.3g}" if isinstance(min_value, float) and not min_value.is_integer() else str(int(min_value))
        formatted_max = f"{max_value:.3g}" if isinstance(max_value, float) and not max_value.is_integer() else str(int(max_value))
        ax.set_title(f"2800 km, min={formatted_min}, max={formatted_max}", fontsize=26)




# # Conditional logic based on depth
#     if depth == 451.5625:  # Check if the depth is exactly 2800 km
#     # Check if a colorbar already exists and remove it
#         if cbar:
#             #cbar.remove()

#         # Calculate symmetric color limits based on field and depth
#             if field == "vs":
#                 min, max = data.min(), data.max()
#             elif field == "vp":
#                 min, max = data.min(), data.max()
#             elif field == "density":
#                 min, max = data.min(), data.max()
#             elif field == "t":
#                 min, max = data.min(), data.max()

#             #ax.set_title(f"{int(depth_label)} km depth, max="(vmax), fontsize=30)
#         ax.get_images()[0].set_clim(min, max)
#         print('===========',max)
#         #ax.set_title(f"{int(depth_label)} km, min={data.min():.3g}, max={data.max():.3g}", fontsize=30)
#         min_value = data.min()
#         max_value = data.max()
#         formatted_min = f"{min_value:.3g}" if isinstance(min_value, float) and not min_value.is_integer() else str(int(min_value))
#         formatted_max = f"{max_value:.3g}" if isinstance(max_value, float) and not max_value.is_integer() else str(int(max_value))
#         ax.set_title(f"451 km, min={formatted_min}, max={formatted_max}", fontsize=26)







    # Get the position of the main axisx
        pos = ax.get_position()
    # Compute colorbar position
        cbar_x = pos.x0 + (pos.width - cbar_length) / 2
        cbar_y = pos.y0 - cbar_pad
    
    # Create a new axis for the colorbar
        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_length, cbar_height])
    # Add the colorbar
        cbar = plt.colorbar(ax.get_images()[0], cax=cbar_ax, orientation='horizontal')
        cbar.set_label(cbar_label, fontsize=24)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.xaxis.label.set_size(20)
        # Customizing the colorbar ticks
        cbar_max = max  # Find the maximum value in the data
        cbar.set_ticks([min, max])  # Set the ticks
        cbar.set_ticklabels(['min', 'max'],fontsize=30)  # Set the labels
    else:
        print("Depth is not 2800 km, skipping colorbar.")

    # Adjust layout to prevent clipping and reduce white space
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)

    # Save the current figure to a file in the current directory
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"Saved depth slice plot to file: {output_filename}")
    plt.close(fig)

def combine_images_with_titles(image_files_vs, image_files_vp, image_files_density, image_files_temperature, output_filename):
    # Load all images
    images_vs = [plt.imread(image_file) for image_file in image_files_vs]
    images_vp = [plt.imread(image_file) for image_file in image_files_vp]
    images_density = [plt.imread(image_file) for image_file in image_files_density]
    images_temperature = [plt.imread(image_file) for image_file in image_files_temperature]
    
    n_rows = len(images_vs)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 4, wspace=0, hspace=0)  # No space between subplots
    
    titles = ["Vs", "Vp", "Density", "Temperature"]
    
    for i, title in enumerate(titles):
        axes[0, i].set_title(title, fontsize=30, fontweight='bold')
        
    
    for row in range(n_rows):
        axes[row, 0].imshow(images_vs[row])
        axes[row, 1].imshow(images_vp[row])
        axes[row, 2].imshow(images_density[row])
        axes[row, 3].imshow(images_temperature[row])
        
        for col in range(4):
            axes[row, col].axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"Saved combined depth slices plot to file: {output_filename}")
    plt.close(fig)

# Load plate boundaries
plate_boundaries = {
    'ridge': load_plate_boundaries('/Users/monikagonciarz/Desktop/Internship/Plate_boundary_files/ridge.gmt'),
    'trench': load_plate_boundaries('/Users/monikagonciarz/Desktop/Internship/Plate_boundary_files/trench.gmt'),
    'transform': load_plate_boundaries('/Users/monikagonciarz/Desktop/Internship/Plate_boundary_files/transform.gmt')
}

# Define basepath and model_name
basepath = "/Volumes/Monika/TERRA_models/"

model_list_file = "/Volumes/Monika/TERRA_models/model_list.txt"
with open(model_list_file, 'r') as f:
    model_names = [line.strip() for line in f if line.strip()]

print(model_names)

# Define parameters for the plot
fields = ["vs", "vp", "density", "t"]  # Fields to plot
#depths = [90.3125, 135.46875]  # Depths in km
depths = [90.3125, 135.46875, 270.9375, 451.5625, 587.03125, 812.8125, 993.4357, 1445.0, 1986.875, 2799.6876]  # Depths in km
radii = [6371 - depth for depth in depths]  # Convert depths to radii
cmap = "coolwarm_r"
cbar_labels = {
    "vs": "Vs [km/s]",
    "vp": "Vp [km/s]",
    "density": "Density [kg/m^3]",
    "t": "Temperature [K]"
}
cbar_height = 0.05  # Increase colorbar thickness (in figure coordinates)
cbar_pad = 0.175    # Adjust padding between plot and colorbar
cbar_length = 0.7   # Adjust length of the colorbar (relative to the figure width)
vmin = None  # Set vmin=None to calculate this automatically
vmax = None  # Set vmax=None to calculate this automatically

for model_name in model_names:
    # Form the full path to the model
    model_path = os.path.join(basepath, model_name)

    # Plot vs, vp, density, and temperature separately and save the images
    image_files_vs = []
    image_files_vp = []
    image_files_density = []
    image_files_temperature = []

    for depth, radius in zip(depths, radii):
        for field in fields:
            output_filename = f"{model_name}_depth_slice_abs_{field}_{int(depth)}.png"
            plot_depth_slice(model_path, field, radius, depth, cmap, cbar_labels[field], cbar_height, cbar_pad, cbar_length, plate_boundaries, vmin, vmax, output_filename)
            if field == "vs":
                image_files_vs.append(output_filename)
            elif field == "vp":
                image_files_vp.append(output_filename)
            elif field == "density":
                image_files_density.append(output_filename)
            elif field == "t":
                image_files_temperature.append(output_filename)

    # Combine the individual images into a single figure with four columns and titles
    combined_output_filename = f"{model_name}_combined_depth_slices_abs.png"
    combine_images_with_titles(image_files_vs, image_files_vp, image_files_density, image_files_temperature, combined_output_filename)

