## Initial Creation by Kevin Phan https://github.com/kevinphaneos
## Modified by David Chaparro

import numpy as np
import cupy as cp
from photutils.psf import IntegratedGaussianPRF
import matplotlib.pyplot as plt
import png
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astrometry import PositionHint
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry
import astrometry
import astroalign as aa
from astropy.table import QTable

def detect_stars(stacked_frames: np.ndarray, detection_threshold:float=0.5, contrast:float=0.5) -> list[float,float]:
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

def match_to_catalogue(extracted_stars, ra_hint:float=0, dec_hint:float=0, rad_hint:float=0, scales:set={4, 5} ):
    '''
    Matches a list of stars to a skyfield, allowing for an astrometric fit.

    Note: Scales 4 to 5 are required for current SatSim configs.

    Input: List (x, y) coordinates of stars.

    Out: Astrometric solution from astrometry.net if successful, None otherwise.
    '''
    solver = astrometry.Solver(
        astrometry.series_5200.index_files(
            cache_directory= '/mnt/c/Documents and Settings/david.chaparro/Documents/Astrometry/portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE',
            scales = scales,
        )
    )

    solution = solver.solve(
        stars=extracted_stars,
        size_hint=None,
        position_hint=PositionHint(ra_hint,dec_hint,rad_hint),
        solution_parameters=astrometry.SolutionParameters(),
    )
    if solution.has_match():
        return solution
    return None

def skycoord_to_pixels(astrometric_solution):
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
    import astrometric_localization as star_trak
    import matplotlib.pyplot as plt
    import numpy as np

    from astropy.io import fits
    from astropy.utils.data import get_pkg_data_filename
    from astropy.visualization import ZScaleInterval
    from astropy.wcs import WCS

    path = get_pkg_data_filename("sat_00000.0000.fits")
    fits_file = fits.open(path)[0]

    header = fits_file.header
    data = fits_file.data

    star_locations = star_trak.detect_stars(data, detection_threshold=0.99, contrast=.25)
    star_x = [lox[0] for lox in star_locations]
    star_y = [lox[1] for lox in star_locations]

    # Visualize the data using matplotlib
    # plt.figure(figsize=(10, 10))
    # plt.imshow(data, cmap='gray', origin='lower')
    # plt.plot(star_x, star_y, '.', color='r', alpha = .2)
    # plt.colorbar()
    # plt.title('FITS Image: Horsehead Nebula')
    # plt.show()

    scalar = ZScaleInterval(contrast=.25)
    scaled_data = scalar(data)

    solution = star_trak.match_to_catalogue(star_locations)
    print(solution)