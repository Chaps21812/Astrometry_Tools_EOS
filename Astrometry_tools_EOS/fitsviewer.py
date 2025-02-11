# Import necessary libraries
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.io.fits.header import Header
from typing import Optional, Union
from astrometry import Solution

def view_fits(input_data:str, stars:Optional[list]=None, solution_stars:Solution=None, image:np.ndarray=None, print_header:bool=False, show_image:bool=True) -> tuple[Header,np.ndarray]:
    
    # Load a sample FITS file from astropy's tutorial data
    # file_path = get_pkg_data_filename(path)
    fits_file = fits.open(input_data)  # Open the FITS file
    hdu = fits_file[0]  # Primary HDU
    header = hdu.header
    data = hdu.data
    name = input_data.split("/")[-1]

    # Display header information
    if print_header:
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

if __name__ == "__main__":
    view_fits("/mnt/c/Users/david.chaparro/My Documents/Astrometry/modified_horsehead.fits")