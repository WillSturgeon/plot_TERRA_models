import os
import numpy as np
import terratools
from terratools.terra_model import read_netcdf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pdf2image import convert_from_path
from PIL import Image
import geopy.distance

def great_circle_points(start_lat, start_lon, azimuth, distance_km=20020, num_points=100):
    """
    Calculate points along a great circle path starting from a given point and azimuth.

    Parameters:
    - start_lat (float): Starting latitude.
    - start_lon (float): Starting longitude.
    - azimuth (float): Azimuth in degrees from North.
    - distance_km (float): Total distance in kilometers to cover along the great circle.
    - num_points (int): Number of points to calculate along the path.

    Returns:
    - lats (list of float): Latitudes of the points.
    - lons (list of float): Longitudes of the points.
    """
    lats = []
    lons = []
    
    for fraction in np.linspace(0, 1, num_points):
        intermed = geopy.distance.distance(kilometers=distance_km * fraction).destination((start_lat, start_lon), azimuth)
        lats.append(intermed.latitude)
        lons.append(intermed.longitude)
        
    return lats, lons

def round_to_nearest(value, base):
    """
    Round a value to the nearest multiple of a base.

    Parameters:
    - value (float): The value to round.
    - base (float): The base to round to.

    Returns:
    - rounded_value (float): The rounded value.
    """
    return base * round(value / base)

def create_plot(model, parameter, cbar_label, filename, lon, lat, azimuth, reverse_cmap=False):
    """
    Create and save an absolute value plot for a specified parameter with dynamic colorbar limits.

    Parameters:
    - model: The terratools model object.
    - parameter (str): The parameter name to plot.
    - cbar_label (str): Label for the colorbar.
    - filename (str): Output filename for the plot PDF.
    - lon (float): Longitude for the plot section.
    - lat (float): Latitude for the plot section.
    - azimuth (float): Azimuth in degrees from North for the plot section.
    - reverse_cmap (bool): Whether to reverse the colormap.

    Returns:
    - None
    """
    print(f"Creating plot for {parameter}")
    print(f"Location: Latitude={lat}, Longitude={lon}, Azimuth={azimuth} degrees from North")

    # Determine colormap
    cmap = "coolwarm" if reverse_cmap else "coolwarm_r"

    # Adjust azimuth for model.plot_section()
    adjusted_azimuth = azimuth 
    print(f"Adjusted azimuth for model.plot_section(): {adjusted_azimuth} degrees")

    # Plot the absolute field
    fig, ax, cbar = model.plot_section(
        parameter,
        lon=lon,
        lat=lat,
        azimuth=adjusted_azimuth,
        distance=181,
        minradius=6370 - 2800,
        delta_radius=10,
        method="triangle",
        return_cbar=True,
        show=False  # Set show=False to prevent interactive display
    )

    # Extract the section data from the plot
    section_data = np.concatenate([coll.get_array().data for coll in ax.collections])
    
    # Calculate vmin and vmax from the section data
    vmin = np.min(section_data)
    vmax = np.max(section_data)
    print(f"Calculated vmin={vmin} and vmax={vmax} from section data")

    # Round vmin and vmax to the nearest sensible values (e.g., 0.5)
    vmin = round_to_nearest(vmin, 0.5)
    vmax = round_to_nearest(vmax, 0.5)
    print(f"Rounded vmin={vmin} and vmax={vmax}")

    # Normalize the color scale
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Apply the colormap and normalization to each collection in the plot
    for collection in ax.collections:
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_clim(vmin, vmax)
        print(f"Set collection color limits to vmin={vmin}, vmax={vmax}")

    # Remove the automatically generated colorbar
    if cbar:
        cbar.remove()

    # Create a new ScalarMappable for the colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create a new axis for the colorbar
    cbar_height = 0.02  # Colorbar thickness (in figure coordinates)
    cbar_pad = -0.05     # Padding between plot and colorbar
    ax_pos = ax.get_position()
    cbar_length = 0.55   # Length of the colorbar to match the width of the axis

    # Calculate the absolute position of the colorbar in figure coordinates
    cbar_x = ax_pos.x0
    cbar_y = ax_pos.y0 - cbar_pad - cbar_height  # Adjust the y-position with padding

    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_length, cbar_height])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label(cbar_label)  # Add label to colorbar

    # Customize colorbar properties
    cbar.ax.tick_params(labelsize=12)  # Change the size of the colorbar tick labels
    cbar.ax.xaxis.label.set_size(14)   # Change the size of the colorbar label

    # Adjust the colorbar ticks to be uniformly spaced
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)

    # Optionally convert density ticks to g/cm³ if applicable
    if parameter == "density":
        cbar.ax.set_xticklabels([f"{tick / 1000:.2f}" for tick in ticks])  # Example conversion

    print(f"Set colorbar ticks to {ticks}")
    print(f"Set colorbar limits to vmin={vmin}, vmax={vmax}")

    # Enable and customize the grid lines for y-axis only
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='k')
    ax.xaxis.grid(False)  # Turn off x-axis grid lines
    ax.set_yticks([6370 - 410, 6370 - 660, 6370 - 1000])
    ax.set_yticklabels([])  # Remove depth labels

    # Remove specific depth labels on the x-axis
    ax.set_xticklabels([])  # Remove x-axis tick labels

    # Remove the major and minor ticks on the x-axis
    ax.tick_params(axis='x', which='both', length=0)  # This removes all x-axis ticks

    # Remove all white space
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Plot circles at the top of the cross-section corresponding to inset map colors
    xticks = ax.get_xticks()
    circle_y_position = 6371  # Set to the radius for the surface
    circle_colors = ['green'] + ['white'] * 5 + ['red']
    for i, tick in enumerate(xticks):
        color = circle_colors[i % len(circle_colors)]
        ax.annotate('o', xy=(tick, circle_y_position), xycoords='data',
                    xytext=(0, -0), textcoords='offset points', ha='center', va='center',
                    fontsize=5, color=color, bbox=dict(boxstyle="circle,pad=0.3", edgecolor='black', facecolor=color))

    # Save the current figure to a file
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved {parameter} plot to file: {filename}")
    plt.close(fig)

    # Convert the PDF to an image, crop it, and save the cropped image as a new PDF
    crop_image(filename)

def crop_image(image_path):
    """
    Crop the top and bottom white spaces from a PDF and save the cropped image as a new PDF.
    
    Parameters:
    - image_path (str): Path to the original PDF file.
    
    Returns:
    - None
    """
    # Convert PDF to image
    try:
        images = convert_from_path(image_path, dpi=300)
    except Exception as e:
        print(f"Error converting {image_path}: {e}, using placeholder.")
        images = []
    
    for img in images:
        width, height = img.size
        # Define the bounding box for cropping (only top and bottom)
        left = 0
        top = 340
        right = width
        bottom = height - 50
        bbox = (left, top, right, bottom)
        cropped_img = img.crop(bbox)
        cropped_image_path = image_path.replace('.pdf', '_cropped.png')
        cropped_img.save(cropped_image_path)
        # Convert cropped image back to PDF
        cropped_img.save(image_path.replace('.pdf', '_cropped.pdf'), 'PDF', resolution=100.0)
        os.remove(cropped_image_path)

def create_inset_map(start_lon, start_lat, azimuth, plate_boundaries, output_filename):
    """
    Creates an inset map with the Robinson projection, plots plate boundaries,
    and marks the starting point with an azimuth line. The map is centered on the midpoint
    of the great circle path.

    Parameters:
    - start_lon (float): Starting longitude for the map.
    - start_lat (float): Starting latitude for the map.
    - azimuth (float): Azimuth in degrees from North to plot the great circle.
    - plate_boundaries (dict): Dictionary containing plate boundary types and their coordinates.
    - output_filename (str): Path to save the generated inset map PDF.
    
    Returns:
    - None
    """
    # Calculate great circle points
    lats, lons = great_circle_points(start_lat, start_lon, azimuth, num_points=100)
    
    # Find the midpoint
    midpoint_index = len(lats) // 2
    midpoint_lat = lats[midpoint_index]
    midpoint_lon = lons[midpoint_index]
    print(f"Midpoint at latitude={midpoint_lat}, longitude={midpoint_lon}")

    # Initialize the Robinson projection centered on the midpoint longitude
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.Robinson(central_longitude=midpoint_lon)})
    ax.set_global()

    # Add land and ocean features
    terrain = cfeature.NaturalEarthFeature('physical', 'land', '110m',
                                          edgecolor='face', facecolor=cfeature.COLORS['land'])
    ax.add_feature(terrain)
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines(linewidth=0.5)  # Thinner coastlines for better clarity

    # Remove grid lines for a cleaner map
    ax.gridlines(draw_labels=False, color='none')

    # Plot plate boundaries, excluding any problematic segments
    problematic_segments = {
        'ridge': [72, 99],
        'trench': [110],
        'transform': []
    }

    for boundary_type, segments in plate_boundaries.items():
        for i, boundary in enumerate(segments):
            if i not in problematic_segments.get(boundary_type, []):
                ax.plot(boundary[:, 0], boundary[:, 1], 'black', linewidth=1,
                        transform=ccrs.PlateCarree())

    # Add circles on the inset map at 30-degree intervals along the great circle
    circle_intervals = range(0, 181, 30)  # 0 to 180 degrees inclusive, step of 30
    colors = ['green'] + ['white'] * 5 + ['red']  # Colors for each interval
    circle_lats = []
    circle_lons = []

    for i, interval in enumerate(circle_intervals):
        if interval == 0:
            # Starting point
            point = geopy.distance.distance(kilometers=0).destination((start_lat, start_lon), azimuth)
        else:
            # Each interval corresponds to approximately 30 degrees on Earth's surface (~3339.6 km)
            distance_km = 111.32 * interval
            point = geopy.distance.distance(kilometers=distance_km).destination((start_lat, start_lon), azimuth)
        
        circle_lats.append(point.latitude)
        circle_lons.append(point.longitude)
        print(f"Circle {i} at latitude={point.latitude}, longitude={point.longitude}")
        
        # Plot the circle marker
        ax.scatter(point.longitude, point.latitude, marker='o', s=100,
                   color=colors[i % len(colors)], edgecolor='black',
                   transform=ccrs.PlateCarree(), zorder=10)

    # Draw lines connecting the circles to represent the great circle path
    for j in range(len(circle_lats) - 1):
        ax.plot([circle_lons[j], circle_lons[j + 1]],
                [circle_lats[j], circle_lats[j + 1]],
                'k-', linewidth=1.0, transform=ccrs.Geodetic())

    # Save the inset map to a PDF file
    fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.close(fig)
    print(f"Saved inset map to file: {output_filename}")

def load_plate_boundaries(file_path):
    """
    Load plate boundary data from a .gmt file.
    
    Parameters:
    - file_path (str): Path to the .gmt plate boundary file.
    
    Returns:
    - boundaries (list of np.ndarray): List of boundary segments as NumPy arrays.
    """
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
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    current_boundary.append([lon, lat])
        if current_boundary:
            boundaries.append(np.array(current_boundary))
    return boundaries

# Define base paths
basepath = "/Volumes/Monika/TERRA_models_Franck_adiabat/convert_adiabat"

model_list_file = "/Volumes/Monika/TERRA_models_Franck_adiabat/convert_adiabat/model_list2.txt"
with open(model_list_file, 'r') as f:
    model_names = [line.strip() for line in f if line.strip()]

# Load plate boundaries
plate_boundaries = {
    'ridge': load_plate_boundaries('/Volumes/Monika/Plate_boundary_files/ridge.gmt'),
    'trench': load_plate_boundaries('/Volumes/Monika/Plate_boundary_files/trench.gmt'),
    'transform': load_plate_boundaries('/Volumes/Monika/Plate_boundary_files/transform.gmt')
}

for model_name in model_names:
    # Form the full path to the model
    model_path = os.path.join(basepath, model_name)

    # Read in the model
    model = read_netcdf([model_path])
    print(f'Plotting model {model_name}')

    # Define color limits for consistent range
    # These are now optional and can be set to None to allow dynamic calculation
    # Alternatively, you can remove these if you want purely dynamic limits
    # Here, we proceed with purely dynamic limits by not using fixed vmin and vmax
    # If you want to set fixed limits, you can uncomment and set the following:
    # global_vmin_vs = 0
    # global_vmax_vs = 10
    # global_vmin_vp = 0
    # global_vmax_vp = 15
    # global_vmin_density = 0
    # global_vmax_density = 5
    # global_vmin_temperature = 0
    # global_vmax_temperature = 100

    # Since we are implementing dynamic limits, we won't set fixed vmin and vmax
    # Print statements are removed for clarity

    # Create and save the plots for the three directions
    # Column 1: Starting point -88.0, -50.0 with Azimuth 117.6299 degrees
    vs_filename_1 = create_plot(
        model=model,
        parameter="vs",
        cbar_label="Vs (km/s)",  # Updated label for absolute values
        filename=f"{model_name}_vs_figure_1.pdf",
        lon=-88.0,
        lat=-50.0,
        azimuth=117.6299,
        reverse_cmap=False  # coolwarm_r
    )
    vp_filename_1 = create_plot(
        model=model,
        parameter="vp",
        cbar_label="Vp (km/s)",  # Updated label for absolute values
        filename=f"{model_name}_vp_figure_1.pdf",
        lon=-88.0,
        lat=-50.0,
        azimuth=117.6299,
        reverse_cmap=False  # coolwarm_r
    )
    density_filename_1 = create_plot(
        model=model,
        parameter="density",
        cbar_label="Density (g/cm³)",  # Updated label for absolute values
        filename=f"{model_name}_density_figure_1.pdf",
        lon=-88.0,
        lat=-50.0,
        azimuth=117.6299,
        reverse_cmap=False  # coolwarm_r
    )
    temperature_filename_1 = create_plot(
        model=model,
        parameter="t",
        cbar_label="Temperature (°C)",  # Updated label for absolute values
        filename=f"{model_name}_temperature_figure_1.pdf",
        lon=-88.0,
        lat=-50.0,
        azimuth=117.6299,
        reverse_cmap=True  # coolwarm
    )
    
    # Column 2: Starting point 137.0, 37.0 with Azimuth 90 degrees
    vs_filename_2 = create_plot(
        model=model,
        parameter="vs",
        cbar_label="Vs (km/s)",  # Updated label for absolute values
        filename=f"{model_name}_vs_figure_2.pdf",
        lon=137.0,
        lat=37.0,
        azimuth=90.0,
        reverse_cmap=False  # coolwarm_r
    )
    vp_filename_2 = create_plot(
        model=model,
        parameter="vp",
        cbar_label="Vp (km/s)",  # Updated label for absolute values
        filename=f"{model_name}_vp_figure_2.pdf",
        lon=137.0,
        lat=37.0,
        azimuth=90.0,
        reverse_cmap=False  # coolwarm_r
    )
    density_filename_2 = create_plot(
        model=model,
        parameter="density",
        cbar_label="Density (g/cm³)",  # Updated label for absolute values
        filename=f"{model_name}_density_figure_2.pdf",
        lon=137.0,
        lat=37.0,
        azimuth=90.0,
        reverse_cmap=False  # coolwarm_r
    )
    temperature_filename_2 = create_plot(
        model=model,
        parameter="t",
        cbar_label="Temperature (°C)",  # Updated label for absolute values
        filename=f"{model_name}_temperature_figure_2.pdf",
        lon=137.0,
        lat=37.0,
        azimuth=90.0,
        reverse_cmap=True  # coolwarm
    )

    # Column 3: Starting point 122.0, 10.0 with Azimuth 90 degrees
    vs_filename_3 = create_plot(
        model=model,
        parameter="vs",
        cbar_label="Vs (km/s)",  # Updated label for absolute values
        filename=f"{model_name}_vs_figure_3.pdf",
        lon=122.0,
        lat=10.0,
        azimuth=90.0,
        reverse_cmap=False  # coolwarm_r
    )
    vp_filename_3 = create_plot(
        model=model,
        parameter="vp",
        cbar_label="Vp (km/s)",  # Updated label for absolute values
        filename=f"{model_name}_vp_figure_3.pdf",
        lon=122.0,
        lat=10.0,
        azimuth=90.0,
        reverse_cmap=False  # coolwarm_r
    )
    density_filename_3 = create_plot(
        model=model,
        parameter="density",
        cbar_label="Density (g/cm³)",  # Updated label for absolute values
        filename=f"{model_name}_density_figure_3.pdf",
        lon=122.0,
        lat=10.0,
        azimuth=90.0,
        reverse_cmap=False  # coolwarm_r
    )    
    temperature_filename_3 = create_plot(
        model=model,
        parameter="t",
        cbar_label="Temperature (°C)",  # Updated label for absolute values
        filename=f"{model_name}_temperature_figure_3.pdf",
        lon=122.0,
        lat=10.0,
        azimuth=90.0,
        reverse_cmap=True  # coolwarm
    )

    # Create and save the inset maps centered on the midpoint of the great circle
    inset_map_1 = create_inset_map(
        start_lon=-88.0,
        start_lat=-50.0,
        azimuth=117.6299,
        plate_boundaries=plate_boundaries,
        output_filename=f"{model_name}_inset_map_1.pdf"
    )
    inset_map_2 = create_inset_map(
        start_lon=137.0,
        start_lat=37.0,
        azimuth=90.0,
        plate_boundaries=plate_boundaries,
        output_filename=f"{model_name}_inset_map_2.pdf"
    )
    inset_map_3 = create_inset_map(
        start_lon=122.0,
        start_lat=10.0,
        azimuth=90.0,
        plate_boundaries=plate_boundaries,
        output_filename=f"{model_name}_inset_map_3.pdf"
    )

    # List of expected files
    expected_files = [
        f"{model_name}_inset_map_1.pdf",
        f"{model_name}_inset_map_2.pdf",
        f"{model_name}_inset_map_3.pdf",
        f"{model_name}_vs_figure_1_cropped.pdf",
        f"{model_name}_vs_figure_2_cropped.pdf",
        f"{model_name}_vs_figure_3_cropped.pdf",
        f"{model_name}_vp_figure_1_cropped.pdf",
        f"{model_name}_vp_figure_2_cropped.pdf",
        f"{model_name}_vp_figure_3_cropped.pdf",
        f"{model_name}_density_figure_1_cropped.pdf",
        f"{model_name}_density_figure_2_cropped.pdf",
        f"{model_name}_density_figure_3_cropped.pdf",
        f"{model_name}_temperature_figure_1_cropped.pdf",
        f"{model_name}_temperature_figure_2_cropped.pdf",
        f"{model_name}_temperature_figure_3_cropped.pdf",
    ]

    # Check for missing files and use placeholders
    images = []
    for pdf_file in expected_files:
        if os.path.exists(pdf_file):
            try:
                pages = convert_from_path(pdf_file, 300)
                images.append(pages[0])
            except Exception as e:
                print(f"Error converting {pdf_file}: {e}, using placeholder.")
                images.append(np.ones((2550, 3300, 3), dtype=np.uint8) * 255)  # Placeholder white image
        else:
            print(f"File not found: {pdf_file}, using placeholder.")
            images.append(np.ones((2550, 3300, 3), dtype=np.uint8) * 255)  # Placeholder white image

    # Create a new figure for combining the images in a 5x3 grid
    fig_combined, axes = plt.subplots(5, 3, figsize=(21, 24), constrained_layout=True)

    # Order of the plots
    ordered_images = [
        images[0], images[1], images[2],      # inset maps
        images[3], images[6], images[9],      # vs
        images[4], images[7], images[10],     # vp
        images[5], images[8], images[11],     # density
        images[12], images[13], images[14]    # temperature
    ]

    # Add the images to the grid in the correct order
    for i, ax in enumerate(axes.flat):
        if i < len(ordered_images):
            print(f"Adding image {i+1}/{len(ordered_images)}")
            ax.imshow(ordered_images[i])
        else:
            print(f"No image for index {i}, adding placeholder.")
            ax.imshow(np.ones((2550, 3300, 3), dtype=np.uint8) * 255)  # Placeholder white image
        ax.axis('off')  # Turn off axis

    # Save the combined figure to a file
    combined_output_filename = f"{model_name}_combined_absolute_cross_section.pdf"
    fig_combined.savefig(combined_output_filename, bbox_inches='tight', pad_inches=0)
    print(f"Saved combined cross-section plot to file: {combined_output_filename}")

print("Script finished.")
