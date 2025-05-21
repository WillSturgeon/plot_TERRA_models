# Plotting (seismically converted) TERRA models, as part of the MC^2 project.
Scripts for plotting TERRA models. Depth slices and cross sections. Absolute Vs, Vp, density and perturbations from glboal mean.  

1. plot_TERRA_xsections_perts.py  
This script plots perturbations relative to the 1D (whole mantle) average at each depth (i.e., the global mean profile at each radius/depth), not just the average along the cross-section.
You will need to change the "basepath" path, "model_list_file" path, the plate boundary paths and change the lat, lon coordinates.
** Note that the lat, lon define the bottom left of the cross-section, not the centre.

2. plot_TERRA_xsections_abs.py   
Same as the scvript above, but it plots absolute values instead of perturbations.

3. plot_TERRA_depths_slices_pertsfromav.py  
This script plots depth slices in terms of perutbations from the average at each depth. The depths and limits of the colorbar are defined in this section (e.g. for Vs):  
       if field == "vs":  
        if int(depth_label) == 90:  
            vmin, vmax = -5, 5  
        elif int(depth_label) == 135:  
            vmin, vmax = -5, 5  
        elif int(depth_label) == 270:  
            vmin, vmax = -2, 2  
        elif int(depth_label) == 451:  
            vmin, vmax = -2, 2  
        elif int(depth_label) == 587:  
            vmin, vmax = -2, 2  
        elif int(depth_label) == 812:  
            vmin, vmax = -1, 1  
        elif int(depth_label) == 993:  
            vmin, vmax = -1, 1  
        elif int(depth_label) == 1445:  
            vmin, vmax = -1, 1  
        elif int(depth_label) == 1986:  
            vmin, vmax = -1, 1  
        elif int(depth_label) == 2800:  
            vmin, vmax = -1, 1  

4. plot_TERRA_depth_slices_absolute.py  
Same as the script above, but it plots absolute values.
