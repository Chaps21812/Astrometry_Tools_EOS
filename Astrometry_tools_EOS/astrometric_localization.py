## Initial Creation by Kevin Phan https://github.com/kevinphaneos
## Modified by David Chaparro

import numpy as np
import cupy as cp
from photutils.psf import IntegratedGaussianPRF
import matplotlib.pyplot as plt
import png
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astrometry import PositionHint, SizeHint
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry
import astrometry
import astroalign as aa
from astropy.table import QTable
from astropy.coordinates import SkyCoord
from astrometry import Solution
from typing import Optional, List
from astropy.wcs import WCS

def detect_stars(stacked_frames: np.ndarray, detection_threshold:float=0.5, contrast:float=0.5) -> np.ndarray:
    '''
    Utilizes a integrated gaussian point spread function to identify stars.

    Notes: Current configuration of the PSF model works for a scale of 4-5 arcmin
           sized image. Will make this more adjustable with calculations if needed.

    Input: The stacked frames to be processed for astrometric localization. Works
           best when background has already been corrected.

    Output: A numpy array of shape (2, N) where N is the number of stars extracted. 
            Contains the x and y pixel coordinates of each extracted star.
    '''
    scalar = ZScaleInterval(contrast=contrast)
    scaled_data = scalar(stacked_frames)
    percentile_raw_value = np.percentile(scaled_data,detection_threshold*100)

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (11, 11)
    finder = DAOStarFinder(percentile_raw_value, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,aperture_radius=4)
    phot = psfphot(scaled_data)
    stars = np.asarray([[x, y] for x, y in zip(phot['x_fit'], phot['y_fit'])])
    return stars

def match_to_catalog_scale_search(extracted_stars:list, header:dict=None) -> Optional[Solution]:
    """
    Searches all astrometric scales to find starfield matches. Once it finds a solution, it stops the search. 

    Args:
        extracted_stars (list): List of the extracted stars in an List of (x, y) coordinates of stars.
        header (str): FITS file header

    Returns:
        bool: Description of what the function returns.
    """
    for i in range(6):
        match = match_to_catalogue(extracted_stars, header=header, scales={i+1})
        if match is not None:
            print("Match found at scale: {}".format(i+1))
            return match
        print("No match found at scale: {}".format(i+1))
    return None

def match_to_catalogue(extracted_stars:list, header:dict=None, scales:set={4,5}):
    '''
    Matches a list of stars to a skyfield, allowing for an astrometric fit.

    Note: Scales 4 to 5 are required for current SatSim configs.

    Input: List (x, y) coordinates of stars.

    Out: Astrometric solution from astrometry.net if successful, None otherwise.
    '''
    solver = astrometry.Solver(
        astrometry.series_5200.index_files(
            cache_directory= '/mnt/c/Users/david.chaparro/My Documents/Astrometry/portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE',
            scales = scales,
        )
    )

    if header is not None:
        # Initialize WCS from the header
        wcs = WCS(header)

        # Example: Convert pixel coordinates to celestial coordinates
        corner1_pixel = [0, 0]  # Example pixel coordinates (x, y)
        corner2_pixel = [-1, -1]  # Example pixel coordinates (x, y)
        corner3_pixel = [0, -1]  # Example pixel coordinates (x, y)
        ra_dec1 = wcs.pixel_to_world(corner1_pixel[0], corner1_pixel[1])
        ra_dec2 = wcs.pixel_to_world(corner2_pixel[0], corner2_pixel[1])
        ra_dec3 = wcs.pixel_to_world(corner3_pixel[0], corner3_pixel[1])
        r1 = ra_dec1.ra.degree
        d1 = ra_dec1.dec.degree
        r2 = ra_dec2.ra.degree 
        d2 = ra_dec2.dec.degree 
        r3 = ra_dec3.ra.degree 
        d3 = ra_dec3.dec.degree 
        pi_conv = np.pi/180
        deg_conv = 180/np.pi
        size_hint = deg_conv*np.arccos(np.sin(d1*pi_conv)*np.sin(d2*pi_conv)+np.cos(d1*pi_conv)*np.cos(d2*pi_conv)*np.cos((r1-r2)*pi_conv))*3600
        search_radius = 5

        print("Unknown hints")
        print("radius Hint: {}".format(search_radius))
        print("size lower Hint: {}".format(size_hint*.1))
        print("size upper Hint: {}".format(size_hint*2))

        hint = PositionHint(r1,d1,search_radius)
        size_hint = SizeHint(size_hint*.1, size_hint*2)
    else:
        hint=None
        size_hint = None

    solution = solver.solve(
        stars=extracted_stars,
        size_hint=size_hint,
        position_hint=hint,
        solution_parameters=astrometry.SolutionParameters(),
    )
    if solution.has_match():
        return solution
    return None

def skycoord_to_pixels(astrometric_solution:Solution) -> List:
    '''
    Converts the solution's sky coordinates to pixel coordinates.

    Input: Astrometric solution

    Output: List of (x, y) coordinates of stars.
    '''
    wcs = astrometric_solution.best_match().astropy_wcs()
    pixels = wcs.all_world2pix([[star.ra_deg, star.dec_deg] for star in astrometric_solution.best_match().stars], 0,)
    return pixels

def solution_to_image_coords(image_stars, solution_stars):
    '''
    The astrometric solution pixel coordinates do not always match with the provided image due to instrument
    calibration differences. Out of the initial extracted stars, this function returns the list of stars that
    do match with the astrometric solution using a transformation matrix. Model is created using sklearns 
    transform model within astroalign.

    Inputs: The image star list of (x, y) coords and the astrometric solution list of (x, y) coords

    Output: Transformation Matric and matched stars in the form of (x, y) coords
    '''
    transf, (s_list, t_list) = aa.find_transform(image_stars, solution_stars)
    dst_calc = aa.matrix_transform(solution_stars, transf.inverse)
    return (transf, dst_calc)

def apply_starmask(s_list, stacked_image, images):
    '''
    Based on the inputted (x, y) coordinates, creates a mask which eliminates known stars from the image

    Input: List of (x, y) coordinates of stars

    Output: A mask that removes stars
    '''
    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (11, 11)
    finder = DAOStarFinder(100.0, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                        aperture_radius=4)
    init_params = QTable()
    init_params['x'] = s_list[:,0] 
    init_params['y'] = s_list[:,1] 
    phot = psfphot(cp.ndarray.get(stacked_image), init_params = init_params)

    processed_images = [psfphot.make_residual_image(cp.ndarray.get(file), (9, 9)).clip(min=0) for file in images]
    return processed_images

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from astropy.io import fits
    from astropy.utils.data import get_pkg_data_filename
    from astropy.visualization import ZScaleInterval
    from astropy.wcs import WCS
    # path = get_pkg_data_filename("Astrometry_tools_EOS/modified_horsehead.fits")
    path = "Astrometry_tools_EOS/modified_horsehead.fits"
    fits_file = fits.open(path)[0]
    header = fits_file.header
    data = fits_file.data

    star_locations = detect_stars(data, detection_threshold=0.90, contrast=.25)
    scalar = ZScaleInterval(contrast=.25)
    scaled_data = scalar(data)

    solution = match_to_catalogue(star_locations, header, scales={1,2,3})
    # solution = match_to_catalogue(star_locations)
    print("Found Solution: ")
    print(solution)
