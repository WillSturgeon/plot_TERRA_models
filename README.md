# plot_TERRA_models
used for plotting TERRA models. Depth slices and cross sections. Absolute Vs, Vp, density and perturbations from mean



plot_TERRA_xsections_perts.py
This script plots perturbations relative to the 1D (whole mantle) average at each depth (i.e., the global mean profile at each radius/depth), not just the average along the cross-section.

You will need to change the "basepath" path, "model_list_file" path the plate boundary paths.

# Define base paths - this is where the TERRA seismically converted .nc files are stored
basepath = "/Volumes/Monika/TERRA_models_Franck_adiabat/convert_adiabat"

# this is a list of all TERRA seismically converted .nc models e.g. "256_044_3800_lith_scl/256_044_3800_lith_scl----conv"
# note that it include the model name, and then again with the "----conv" extension.
model_list_file = "/Volumes/Monika/TERRA_models_Franck_adiabat/convert_adiabat/model_list2.txt"
with open(model_list_file, 'r') as f:
    model_names = [line.strip() for line in f if line.strip()]

# Load plate boundaries
plate_boundaries = {
    'ridge': load_plate_boundaries('Plate_boundary_files/ridge.gmt'),
    'trench': load_plate_boundaries('Plate_boundary_files/trench.gmt'),
    'transform': load_plate_boundaries('Plate_boundary_files/transform.gmt')
}
