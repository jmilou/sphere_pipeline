#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:20:51 2018

@author: jmilli
"""

import numpy as np
from astropy.io import fits
import os

def remove_bad_frames_zimpol(filename,badframes_indices,outputfilename=None):
    """
    Function that reads a Zimpol fits file, and rewrites it to disk after 
    removing some bad frames. 
    Input:
        - filename: the complete path+filename of the fits file to modify
        - badframes_indices: the indices of the frames to remove
    """
    hduList = fits.open(filename)
    header = hduList[0].header
    
    calas = hduList[1].data
    calas_header = hduList[1].header

    bartoli = hduList[2].data
    bartoli_header = hduList[2].header
    nbFrames = header['HIERARCH ESO DET NDIT']
    if isinstance(badframes_indices,(list,range)):
        list_frames = np.arange(nbFrames)
        goodframes = [idf for idf in list_frames if idf not in badframes_indices]
        calas_good = calas[goodframes,:,:]
        bartoli_good = bartoli[goodframes,:,:]
        header['HIERARCH ESO DET NDIT'] = len(goodframes)
        bartoli_header['NAXIS3'] = len(goodframes)
        calas_header['NAXIS3'] = len(goodframes)
        print('Bad frames specified (0-based indexing):',badframes_indices)
    else:
        raise Exception('Badframes should be specified with a list.',
                            'Got ',badframes_indices)

    if outputfilename is None:         
        outputfilename =     filename.replace('.fits','_wo_badframes.fits')
    primary_hdu = fits.PrimaryHDU(header=header)
    hdu_calas = fits.ImageHDU(calas_good,header=calas_header)
    hdu_bartoli = fits.ImageHDU(bartoli_good,header=bartoli_header)
    new_hduList = fits.HDUList([primary_hdu,hdu_calas,hdu_bartoli])
    new_hduList.writeto(outputfilename,overwrite=True,output_verify='ignore')

    return 
    
#if __name__ == "__main__":
#    pathRoot='/Volumes/MILOU_1TB_2/HR4796_zimpol_GTO'
#    pathRaw=os.path.join(pathRoot,'raw_science/FastPolarimetry_000')
#    fitsfile='SPHER.2016-05-25T01:14:12.495.fits'
#    remove_bad_frames_zimpol(os.path.join(pathRaw,fitsfile),[0,1])