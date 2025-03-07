from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
from astropy.io import fits
import numpy as np

def adaptive_iqr(path:str, bkg_subtract:bool=True, verbose:bool=False) -> np.ndarray:
    '''
    Performs Log1P contrast enhancement. Searches for the highest contrast image and enhances stars.
    Optionally can perform background subtraction as well

    Notes: Current configuration of the PSF model works for a scale of 4-5 arcmin
           sized image. Will make this more adjustable with calculations if needed.

    Input: The stacked frames to be processed for astrometric localization. Works
           best when background has already been corrected.

    Output: A numpy array of shape (2, N) where N is the number of stars extracted. 
    '''  
    fits_data = fits.open(path)
    hdu = fits_data[0]
    fits_image = hdu.data

    if bkg_subtract:
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(fits_image, (32, 32), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        fits_image = fits_image-bkg.background

    if verbose:
        print("| Percentile | Contrast |")
        print("|------------|----------|")
    best_contrast_score = 0
    best_percentile = 0
    best_image = None
    percentiles=[]
    contrasts=[]

    for i in range(20):
        #Scans image to find optimal subtraction of median
        percentile = 90+0.5*i
        temp_image = fits_image-np.quantile(fits_image, (percentile)/100)
        temp_image[temp_image < 0] = 0
        scaled_data = np.log1p(temp_image)
        #Metric to optimize, currently it is prominence
        contrast = (np.max(scaled_data[scaled_data > 0])+np.mean(scaled_data[scaled_data > 0]))/2-np.median(scaled_data[scaled_data > 0])
        percentiles.append(percentile)
        contrasts.append(contrast)

        if contrast > best_contrast_score*1.05:
            best_contrast_multiplier = i
            best_image = scaled_data.copy()
            best_contrast_score = contrast
            best_percentile = percentile
        if verbose: print("|    {:.2f}   |   {:.2f}   |".format(percentile,contrast))
    if verbose: print("Best percentile): {}".format(best_percentile))
    return best_image

if __name__ == "__main__":
    adaptive_iqr("/mnt/c/Users/david.chaparro/Documents/Repos/SatSim/output/2025-03-04T11-02-18.704584/ImageFiles/sat_00000.0000.fits", bkg_subtract=False)
