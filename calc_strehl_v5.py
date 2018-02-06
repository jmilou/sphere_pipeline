#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:41:34 2017

@author: cpannetier
"""


#import pandas as pd
import os
import numpy as np
import scipy
#from scipy import stats
#from scipy import signal
from astropy.io import fits

import matplotlib.pyplot as plt
import radial_data as rd

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print ("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder =  [ndims - 1] + list(range( ndims - 1 ))

        for i in range( ndims - 2, -1, -1 ):
            
            
            newa = newa.transpose(trorder)
            
            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print ("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None
    

# Zero-padding function
def zero_padding(im, dim):
    '''
    Function which add zeros on the edge of an image to reach the dimension dim that we want.
    INPUT:
        im = np.array LxL
        dim = dimension of the resulting array
    '''
    sz = im.shape[0]
    addpix = (dim-sz)/2.
    if addpix > 0:
        if isinstance(addpix,int):
            im = np.pad(im, addpix, 'constant')
        else: 
            addpix = int(addpix)+1
            im = np.pad(im, addpix, 'constant')
            im = congrid(im, (dim,dim), method='linear')  
    else:
        print('No zero padding but cropping of image')
        if type(addpix) == int:
            rempix = -addpix
            im = im[rempix:-rempix,rempix:-rempix]
        else: 
            addpix = int(addpix)+1
            rempix = -addpix
            im = im[rempix:-rempix,rempix:-rempix]
            im = congrid(im, (dim,dim), method='linear')  
        
    return im


def ftccd(n):
    '''
    Create the transfer function of a CCD matrix of dimension n x n.
    The FT is centered in [n/2,n/2] and normalized to 1. 
 
    - Formule :
    FT          = sinc(a.u) x sinc (b.v) where a x b are pixels dimensions 
    
    - INPUT:
        im = np.array LxL
        dim = dimension of the resulting array
        
    - OUTPUT:
        n x n numpy array corresponding to the normalized transfert function of the CCD
    '''
    # u1, v1, U et V entiers pour gain memoire
    u1 = np.arange(n) - n/2
    
    U = np.outer(np.zeros(n)+1, u1)
    V = np.outer(u1,np.zeros(n)+1)

    FT = np.sinc(((1/n)*U)) * np.sinc(((1/n)*V))
    
    return FT


def mask(r,dim,oc=0):
    '''
    Create a mask of radius r in an array of dimension dim.
    We can add an central obscuration with the parameter oc.
 
    - INPUT:
        r = aperture radius
        dim = dimension of the resulting array
        oc = 0: radius of the central obturation, 0 by default.
        
    - OUTPUT:
        dim x dim numpy array corresponding to the mask.
    '''
    largeur = dim
    centre = float((largeur-1.0)/2)
    x,y = np.meshgrid(np.arange(float(largeur)),np.arange(float(largeur)))   # Tableaux des coordonnees des pixels
    x = x - centre
    y = y - centre

    mask = np.ones((largeur, largeur))
    rho = np.array((np.sqrt(x**2+y**2)) / float(r), dtype=float) # Each pixel is equal to its distance from the center
#    phi = np.array(np.arctan(y, x + (rho == 0.)), dtype=float) # Each pixel is equal to its angle from the x absciss

    mask = mask*((rho <= 1.) & (rho >= oc))

    return mask



def calc_strehl(psf, oversample=1, oc=0, SCALE='LINEAR', FIT=True, POLFIT=2, \
                SUBSTRACT_NOISE=True, VISU=False,VERBOSE=False,ABS=True,\
                PUPIL=False, APO='APO1',LYOT='ALC2', \
                path_lib = os.getcwd()):
    '''
    Calculate the strehl ratio of an psf image.
    We use the absolute value of the otf, meaning that we assume the PSF to be 
    symmetric but doesn't have to be perfectly centered.
    
    - INPUT:
        psf = square numpy array
        oversample = ratio corresponding to Nyquist spatial period over the pixel size. By default, no oversampling.
        oc = 0: radius of the central obturation, 0 by default.
        SCALE = 
        FIT = if True, we perform a polynomial fit of most centered circles mean values 
        to substract the noise contained in the FTO's zero frequency.
        POLFIT = polynom fitting degree
        SUBSTRACT_NOISE = if True, substraction of noise using the frequencies above Nyquist frequency.
                          To use this option, oversample need to be > 1.
        VISU = display the FTO curves.
        VERBOSE = display comments.
        ABS = if True, we use the absolute value of the OTF. So far, it always use abs. The other option has not
        been developed yet.
        PUPIL = Pupil map
        APO = Apodyser map
        LYOT = Stop Lyot map
        path = working path where are routines
        path_lib = path where are the APO, PUPIL, LYOT and other maps.
        
    - OUTPUT:
        - a scalar number corresponding to the strehl ratio
        - if OTF is given, OTF takes the value of the Optical Transfer Function of the psf.
    '''

    dim = psf.shape[0]
    print('The psf size is {0:}px'.format(dim))

    ###############################################
    ####   CALCULATION OF THE FTO OF THE PSF   ####
    ###############################################
    
    otf=np.abs(np.fft.fftshift(np.fft.fft2(psf))) # Optical Transfert Function with zero frequency centered.
    
    otfcorr = np.copy(otf)
    otf = otf/np.max(otf)               # Normalization of the raw OTF

    otfcorr1D = rd.Radial_data(otfcorr)    # Radial_data object containing information such as means or medians 
                                        # of annulus of growing radius and 1 pixel width.
    
    
    if FIT:                             # POLYNOMIAL FITTING
    
        polfit = POLFIT                 # Polynomial degree
        dimfit = polfit+2               # Number of fitting points
        dimfit+=1
        
        u = otfcorr1D.r[1:dimfit] #otfcorr1D.r[1:dimfit] #np.arange(dimfit-1)+1
        u1 = otfcorr1D.r[0:dimfit] #otfcorr1D.r[0:dimfit]  #np.arange(dimfit)
        v = otfcorr1D.mean[1:dimfit]


        if polfit == 1:
            if dimfit <= 2: dimfit = 3
            result = np.polyfit(u, v, polfit)
            intercept = result[-1]
            yfit1 = result[0]*u1 + result[1]
            if VERBOSE:
                print('\nFIT by polynom order 1')
                print('Number of fitted points:', dimfit-1)

        if polfit == 2:
            if dimfit <= 3: dimfit = 4
            result = np.polyfit(u, v, polfit)
            intercept = result[-1]
            yfit1 = result[0]*u1*u1 + result[1]*u1 + result[2]
            if VERBOSE:
                print('\nFIT by polynom order 2')
                print('Number of fitted points:', dimfit-1)

        if polfit == 3:
            if dimfit <= 4: dimfit = 5
            result = np.polyfit(u, v, polfit)
            intercept = result[-1]
            yfit1 = result[0]*u1*u1*u1 + result[1]*u1*u1 + result[2]*u1 + result[3]
            if VERBOSE:
                print('\nFIT by polynom order 3')
                print('Number of fitted points:', dimfit-1)

        if polfit == 4:
            if dimfit <= 5: dimfit = 6
            result = np.polyfit(u, v, polfit)
            intercept = result[-1]
            yfit1 = result[0]*u1*u1*u1*u1 + result[1]*u1*u1*u1 + result[2]*u1*u1 + result[3]*u1 + result[4]
            if VERBOSE:
                print('\nFIT by polynom order 4')
                print('Number of fitted points:', dimfit-1)

        if (intercept >= otfcorr1D.mean[0]):
            print("\n### WARNING - Fit hasn't been done ###")
            print("Noise seems to be less than zero")
            otfcorr = otfcorr/otfcorr[dim//2][dim//2]

        else:
            otfcorr[dim//2][dim//2] = intercept     # Removing of part of the noise
            otfcorr = otfcorr/intercept             # Normalization

    else:
        if VERBOSE:
            print('\nYou chose not to fit')
        otfcorr = otfcorr/np.max(otfcorr)           # Normalization
           
    
    
    
    # Substraction of background noise
    fcoup = (dim-1)/oversample/2                        # Nyquist spatial frequency (Maximum frequency which makes sens)
    print('The cutoff frequency is {0:} px'.format(fcoup))
    otfcorr1D = rd.Radial_data(otfcorr)
    plateau = np.mean(otfcorr1D.mean[(int(fcoup)+1):])  # Calculation of noise mean
    otfcorr = otfcorr - plateau                         # Substraction of noise
    otfcorr = otfcorr/otfcorr[dim//2][dim//2]           # Normalization
    
    
    # Fonction de tranfert du capteur CCD
    ft_ccd = ftccd(dim)
    otfcorr = otfcorr/ft_ccd
    

    # Removing of the frequencies above D/lambda by putting all the corresponding pixels to 0
    tabmask = mask(fcoup, dim)
    otfcorr = otfcorr*tabmask
    
  
    ###############################################
    #### CONSTRUCTION OF THE TELESCOPE'S PUPIL ####
    ###############################################

    rr = float(dim-1)/(2*oversample)/2          # Radius of the pupil in the Fourier domain
    size_pup = (fcoup,fcoup)


    if PUPIL == True:                           # Loading and resizing the pupil
        if VERBOSE:
            print('\nLoading of the pupil')
        fitsfile = fits.open(path_lib+'/ST_VLT_SYNTHETIC.fits') # Loading
        primary = fitsfile[0].data
        tab = primary
        if VERBOSE:
            print('Size of the original file',np.shape(tab))
            print('Flux before resizing:',np.sum(tab)/tab.size)
        tab = congrid(tab, size_pup, method='linear')           # Resizing of the pupil to size_pup dimensions
        if VERBOSE:                                             # Checking that flux is conserved
            print('Flux after resizing:',np.sum(tab)/tab.size)
            print('Size of the pupil',np.shape(tab))
        tab = zero_padding(tab, dim)                            # Add zeros to reach the dimxdim dimension
        if VERBOSE:
            print('Size after zero padding:',np.shape(tab))

    else:                                         # Construction of a simple pupil by ourselves
        if VERBOSE:
            print('\nConstruction of the pupil')
            print('Nyquist spatial frequency:',fcoup, 'pixels/arcsec')
        tab = mask(rr, dim, oc=oc) # Pupil of the telescope


        
    if APO == 'APO1': # Apodizer 1
        if VERBOSE:
            print('\nApodizer 1')
        fitsfile = fits.open(path_lib+'/APO1_SR.fits')                  # Loading
        primary = fitsfile[0].data
        apo_map = primary
        if VERBOSE:
            print('Size of the original file',np.shape(apo_map))
            print('Flux before resizing:',np.sum(apo_map)/apo_map.size)
        apo_sized = congrid(apo_map, size_pup, method='linear')         # Resizing of the pupil to size_pup dimensions
        if VERBOSE:
            print('Flux after resizing:',np.sum(apo_sized)/apo_sized.size)
            print('Size of the pupil',np.shape(apo_sized))
        apo_sized = zero_padding(apo_sized, dim)                        # Add zeros to reach the dimxdim dimension
        if VERBOSE:
            print('Size after zero padding:',np.shape(apo_sized))
        tab *= apo_sized



    elif APO == 'APO2': # Apodizer 2
        if VERBOSE:
            print('\nApodizer 2')
        fitsfile = fits.open(path_lib+'/APO2_SR.fits')
        primary = fitsfile[0].data
        apo_map = primary
        if VERBOSE:
            print('Size of the original file',np.shape(apo_map))
            print('Flux before resizing:',np.sum(apo_map)/apo_map.size)
        apo_sized = congrid(apo_map, size_pup, method='linear')         # Resizing of the pupil to size_pup dimensions
        if VERBOSE:
            print('Flux after resizing:',np.sum(apo_sized)/apo_sized.size) 
            print('Size of the pupil',np.shape(apo_sized))
        apo_sized = zero_padding(apo_sized, dim)                        # Add zeros to reach the dimxdim dimension
        if VERBOSE:
            print('Size after zero padding:',np.shape(apo_sized))
        tab *= apo_sized

    elif VERBOSE:
        print('\nNo spatial filter')

    if LYOT == 'ALC2': # Lyot stop
        if VERBOSE:
            print('\nLyot stop 2')
        fitsfile = fits.open(path_lib+'/ST_ALC2_SR.fits')
        primary = fitsfile[0].data
        lyot_map = primary
        if VERBOSE:
            print('Size of the original file',np.shape(lyot_map))
            print('Flux before resizing:',np.sum(lyot_map)/lyot_map.size)
        lyot_sized = congrid(lyot_map, size_pup, method='linear')       # Resizing of the pupil to size_pup dimensions
        if VERBOSE:
            print('Flux after resizing:',np.sum(lyot_sized)/lyot_sized.size)
            print('Size of the pupil',np.shape(lyot_sized))
        lyot_sized = zero_padding(lyot_sized, dim)                      # Add zeros to reach the dimxdim dimension
        if VERBOSE:
            print('Size after zero padding:',np.shape(lyot_sized))
        tab *= lyot_sized    
    
    elif VERBOSE: print('\nNo Lyot stop')
        
    otfairy = np.array(np.abs(np.fft.fftshift(np.fft.ifft2(abs(np.fft.fft2(tab))**2))), dtype=float)
    otfairy = otfairy/np.max(otfairy)                                   # Normalization
    
    # Removing of the frequencies above D/lambda
    otfairy = otfairy*tabmask
    
    # Calculation of Strehl ratio
    sr = np.sum(otfcorr)/np.sum(otfairy)

    if VISU:
        print('Strehl ratio: ', sr)
        ###############################################
        #######    VISUALISATION OF THE FTOs    #######
        ###############################################    
        
        otfcorr1D=rd.Radial_data(otfcorr)
        otfairy1D=rd.Radial_data(otfairy)
        otf1D = rd.Radial_data(otf)
        
            # example of use 
        plt.figure(0)
        plt.plot(otfcorr1D.r,otfcorr1D.mean, 'k',label='FTO corrected')
        plt.plot(otfairy1D.r,otfairy1D.mean,'r',label='Airy FTO')
        plt.plot(otf1D.r,otf1D.mean,'k--',label='Raw FTO')
        
        plt.xlabel('Value in ADU')
        if SCALE == 'LOG' or SCALE == 'log':
            plt.yscale('log')
        plt.axis([0 ,size_pup[0] ,1e-5 ,1])
        plt.legend()        
        plt.show()

    return sr


if __name__ == "__main__":
### EXAMPLE OF USE ###

    path = '/Users/jmilli/Documents/survey_discs/statistics_for_cyril/'
    path_lib = '/Users/jmilli/Dropbox/lib_idl/strehl'
    cube_psf = fits.getdata(os.path.join(path+'psfall_norm_at_SR.fits'))
    
    D=8. # primary mirror is 8.2m but the pupil is restricted by the 2ndary mirror at 8m
    central_obscuration = 1.2/D  
    print('Central obscuration diameter: {0:.2f}%'.format(central_obscuration*100.))
    pixel_irdis = 0.01225 # pixel scale for IRDIS [arcsec/px]
    irdis_frequency_sampling = 1/pixel_irdis
    
    
    wl = 1588.8e-9
    pixel_nyquist = 0.5*wl/D*180/np.pi*3600. # Nyquist sampling [arcsec/px]
    oversample = pixel_nyquist/pixel_irdis # oversampling factor
    psf = fits.getdata('/Users/jmilli/Documents/SPHERE/psf_star_mag4-6-8/2018-01-27/pipeline/psf_left_cropped.fits')
    calc_strehl(psf, oversample=oversample, oc=central_obscuration, FIT=True,\
            SUBSTRACT_NOISE=True,VISU=True, VERBOSE=True,ABS=True, PUPIL=True, \
            APO='APO1',LYOT='ALC2', path_lib=path_lib)
    
     
    wl = 1.625e-6 # H band
    print('Pupil diameter: {0:.2f}m, wavelength {1:.3f}microns'.format(D,wl*1.e6))
    pixel_nyquist = 0.5*wl/D*180/np.pi*3600. # Nyquist sampling [arcsec/px]
    print('Nyquist sampling: {0:.4f} arcsec/pix'.format(pixel_nyquist))
    print('Irdis sampling: {0:.4f} arcsec/pix'.format(pixel_irdis))
    oversample = pixel_nyquist/pixel_irdis # oversampling factor
    print('Oversampling {0:.3f}   (Nyquist sampling/Irdis sampling)'.format(oversample)) # 1.68379 at H band.
    
    
    strehls=[]
    i=0
    for psf1 in cube_psf:
        print('Image numero', i)
        i+=1
        strehls.append(calc_strehl(psf1, oversample=oversample, oc=central_obscuration, FIT=True,\
                SUBSTRACT_NOISE=True,VISU=False, VERBOSE=True,ABS=True, PUPIL=True, \
                APO='APO1',LYOT='ALC2' ,path_lib=path_lib))
        
    
    #calc_strehl(psf, oversample=oversample, oc=0, OTF=True, FIT=True, SUBSTRACT_NOISE=True, VISU=True,VERBOSE=False,\
    
    
    #psf = primary[27]
    #calc_strehl(psf, oversample=oversample, oc=0, OTF=True, FIT=True, SCALE='log', SUBSTRACT_NOISE=True, \
    #                               VISU=True, VERBOSE=True,ABS=True, PUPIL=True, APO='APO1',LYOT='ALC2',\
    #                               path=path,path_lib=path_lib)
    
    sr = fits.getdata(os.path.join(path,'sr.fits'))
    
    plt.plot(sr, strehls, '*')
    plt.plot([0,1],[0,1])
    
    #fits.writeto('/Users/cpannetier/Documents/strehl/strehls_cyril.fits',np.array(strehls), overwrite=True)



