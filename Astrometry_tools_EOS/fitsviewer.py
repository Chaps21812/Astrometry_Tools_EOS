# Import necessary libraries
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from PIL import Image
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.io.fits.header import Header
from typing import Optional

def view_fits(path:str, stars:Optional[list], print_header:bool=False, show_image:bool=True) -> tuple[Header,np.ndarray]:
    # Load a sample FITS file from astropy's tutorial data
    file_path = get_pkg_data_filename(path)
    fits_file = fits.open(file_path)  # Open the FITS file

    # Access the primary HDU (Header/Data Unit)
    hdu = fits_file[0]  # Primary HDU

    # Extract header and data
    header = hdu.header
    data = hdu.data

    # Display header information
    if print_header:
        print("\nHeader Information:")
        print(repr(header))

    # Visualize the data using matplotlib
    if show_image:
        fig, (ax_plot, ax_table) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

        # ---- PLOT (Top) ----
        im = ax_plot.imshow(data, cmap='gray', origin='lower')
        ax_plot.set_title('FITS Image: {}'.format(path.split("/")[-1]))
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])
        plt.colorbar(im, ax=ax_plot)

        if not stars:
            star_x = [lox[0] for lox in stars]
            star_y = [lox[1] for lox in stars]
            ax_plot.plot(star_x, star_y, '.', color='r', alpha = .2)


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
            cell.set_height(0.2)

        # Hide axes for table subplot
        ax_table.axis('off')

        # Adjust spacing between plot and table
        plt.subplots_adjust(hspace=0.1)  # Increase hspace to move them apart
        plt.show()

    return (header, data)

if __name__ == "__main__":
    view_fits("/mnt/c/Users/david.chaparro/My Documents/Repos/Astrometry_Tools_EOS/Astrometry_tools_EOS/2025-02-07T11-17-58.676932/ImageFiles/sat_00000.0005.fits")