# Import necessary libraries
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.io.fits.header import Header
from astrometry import Solution
import matplotlib.patches as patches
import json

def view_fits(input_data:str, stars:list=None, solution_stars:Solution=None, image:np.ndarray=None, show_header:bool=False, show_image:bool=True) -> tuple[Header,np.ndarray]:
    """
    Plots image of your fits file, along with optional star locations. 

    Args:
        input_data (str): Input directory for your fits file
        stars (Optional[list]): Optinal list of stars to be plotted in red against image List (x,y) tuples
        solution_stars (Optional[list]): Optinal list of known stars to be plotted in green against image List (x,y) tuples
        image (np.ndarray): Optional image to display instead of default FITS image. 
        show_header (bool): Bool to toggle showing fits header. 
        show_image (bool): Bool to turn off showing the image. Can use to just view fits file

    Returns:
        tuple(np.ndarray, dict): Tuple of your image and your header file
    """
    
    # Load a sample FITS file from astropy's tutorial data
    # file_path = get_pkg_data_filename(path)
    fits_file = fits.open(input_data)  # Open the FITS file
    hdu = fits_file[0]  # Primary HDU
    header = hdu.header
    data = hdu.data
    name = input_data.split("/")[-1]

    # Display header information
    if show_header:
        print("\nHeader Information:")
        print(repr(header))

    # Visualize the data using matplotlib
    if show_image:
        fig, (ax_plot, ax_table) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

        # ---- PLOT (Top) ----
        if image is None:
            im = ax_plot.imshow(data, cmap='gray', origin='lower')
        else:
            im = ax_plot.imshow(image, cmap='gray', origin='lower')
        ax_plot.set_title('FITS Image: {}'.format(name))
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])
        plt.colorbar(im, ax=ax_plot)

        if stars is not None:
            star_x = [lox[0] for lox in stars]
            star_y = [lox[1] for lox in stars]
            ax_plot.plot(star_x, star_y, '.', color='r', alpha = .3)
        if solution_stars is not None:
            sol_star_x = [lox[0] for lox in solution_stars]
            sol_star_y = [lox[1] for lox in solution_stars]
            ax_plot.plot(sol_star_x, sol_star_y, '.', color='g', alpha = .3)
            

        # ---- TABLE (Bottom) ----
        row_labels = [ 'Size', 'Date', 'Right Ascension', 'Declination']
        table_data = [ ["{}x{}".format(header["NAXIS1"],header["NAXIS2"])], [header["DATE-OBS"]], [header["OBJCTRA"]], [header["OBJCTDEC"]]]
        table = ax_table.table(cellText=table_data, 
                            rowLabels=row_labels, 
                            cellLoc='center', 
                            loc='center')

        # Format table appearance
        # table.auto_set_font_size(False)
        # table.set_fontsize(10)

        # Adjust row height for better spacing
        for key, cell in table.get_celld().items():
            cell.set_height(0.3)

        # Hide axes for table subplot
        ax_table.axis('off')

        # Adjust spacing between plot and table
        plt.subplots_adjust(hspace=0.1)  # Increase hspace to move them apart
        plt.show()

    return (header, data)

def view_fits_bbox(input_data:str, center:tuple, box_size:tuple, image:np.ndarray=None,) -> tuple[Header,np.ndarray]:
    """
    Plots image of your fits file, along with optional star locations. 

    Args:
        input_data (str): Input directory for your fits file
        stars (Optional[list]): Optinal list of stars to be plotted in red against image List (x,y) tuples
        solution_stars (Optional[list]): Optinal list of known stars to be plotted in green against image List (x,y) tuples
        image (np.ndarray): Optional image to display instead of default FITS image. 
        show_header (bool): Bool to toggle showing fits header. 
        show_image (bool): Bool to turn off showing the image. Can use to just view fits file

    Returns:
        tuple(np.ndarray, dict): Tuple of your image and your header file
    """
    
    # Load a sample FITS file from astropy's tutorial data
    # file_path = get_pkg_data_filename(path)
    fits_file = fits.open(input_data)  # Open the FITS file
    hdu = fits_file[0]  # Primary HDU
    header = hdu.header
    data = hdu.data
    name = input_data.split("/")[-1]

    # Visualize the data using matplotlib
    fig, (ax_plot, ax_table) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

    # ---- PLOT (Top) ----
    if image is None:
        im = ax_plot.imshow(data, cmap='gray', origin='lower')
    else:
        im = ax_plot.imshow(image, cmap='gray', origin='lower')
    ax_plot.plot(center[0],center[1], '.', 'r', alpha=.3)
    # Create a Rectangle patch
    rect = patches.Rectangle((center[0]-int(box_size[0]/2), center[1]-int(box_size[1]/2)), box_size[0], box_size[1], linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
    ax_plot.add_patch(rect)
    ax_plot.set_title('FITS Image: {}'.format(name))
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])
    plt.colorbar(im, ax=ax_plot)

            

    # ---- TABLE (Bottom) ----
    row_labels = [ 'Size', 'Date', 'Right Ascension', 'Declination']
    table_data = [ ["{}x{}".format(header["NAXIS1"],header["NAXIS2"])], [header["DATE-OBS"]], [header["OBJCTRA"]], [header["OBJCTDEC"]]]
    table = ax_table.table(cellText=table_data, 
                        rowLabels=row_labels, 
                        cellLoc='center', 
                        loc='center')

    # Format table appearance
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)

    # Adjust row height for better spacing
    for key, cell in table.get_celld().items():
        cell.set_height(0.3)

    # Hide axes for table subplot
    ax_table.axis('off')

    # Adjust spacing between plot and table
    plt.subplots_adjust(hspace=0.1)  # Increase hspace to move them apart
    plt.show()

    return (header, data)

def view_star_streak(input_data:str, image:np.ndarray=None):
    """
    Plots image of your fits file, along with optional star locations. 

    Args:
        input_data (str): Input directory for your fits file
        stars (Optional[list]): Optinal list of stars to be plotted in red against image List (x,y) tuples
        solution_stars (Optional[list]): Optinal list of known stars to be plotted in green against image List (x,y) tuples
        image (np.ndarray): Optional image to display instead of default FITS image. 
        show_header (bool): Bool to toggle showing fits header. 
        show_image (bool): Bool to turn off showing the image. Can use to just view fits file

    Returns:
        tuple(np.ndarray, dict): Tuple of your image and your header file
    """
    
    # Load a sample FITS file from astropy's tutorial data
    # file_path = get_pkg_data_filename(path)
    fits_file = fits.open(input_data)  # Open the FITS file
    hdu = fits_file[0]  # Primary HDU
    data = hdu.data
    name = input_data.split("/")[-1]
    overall_path = input_data.split("/ImageFiles")[0]
    annotation_path = overall_path+"/Annotations/" + name
    annotation_path = annotation_path.replace(".fits",".json")

    with open(annotation_path) as json_data:
        annotations = json.load(json_data)
        json_data.close()

    (annotations["data"]["objects"])

    # ---- PLOT (Top) ----
    plt.figure(figsize=(12,12))
    plt.title('FITS Image: {}'.format(name))
    if image is None: im = plt.imshow(data, cmap='gray', origin='lower')
    else: im = plt.imshow(image, cmap='gray', origin='lower')

    for object in annotations["data"]["objects"]:
        if object["class_id"] == 1:
            plt.plot(object["x_center"]*256, object["y_center"]*256, '.', color='r', alpha = .3)
        if object["class_id"] == 2:
            linex = [object["x_start"]*256,object["x_end"]*256]
            liney = [object["y_start"]*256,object["y_end"]*256]
            plt.plot(object["x_mid"]*256, object["y_mid"]*256, '.', color='g', alpha = .3)
            plt.plot(linex,liney, '-', color='g', alpha = .3)
            
    # Adjust spacing between plot and table
    plt.subplots_adjust(hspace=0.1)  # Increase hspace to move them apart
    plt.show()

if __name__ == "__main__":
    view_fits("/mnt/c/Users/david.chaparro/My Documents/Astrometry/modified_horsehead.fits")