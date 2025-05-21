# Plotting (seismically converted) TERRA models, as part of the MC^2 project.
Scripts for plotting TERRA models. Depth slices and cross sections. Absolute Vs, Vp, density and perturbations from glboal mean.

1. plot_TERRA_xsections_perts.py
This script plots perturbations relative to the 1D (whole mantle) average at each depth (i.e., the global mean profile at each radius/depth), not just the average along the cross-section.
You will need to change the "basepath" path, "model_list_file" path, the plate boundary paths and change the lat, lon coordinates.
** Note that the lat, lon define the bottom left of the cross-section, not the centre.

2. plot_TERRA_xsections_abs.py
Same as the scvript above, but it plots absolute values instead of perturbations.

3. plot_TERRA_depths_slices_pertsfromav.py

