    # -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:27:05 2015
Modified on Sun Jun 2 to add the function ifs_transmission
@author: jmilli
"""
from astropy.io import ascii,fits
import numpy as np
import os #,sys
from scipy.interpolate import interp1d
#from photutils.morphology import centroid_2dg
from fit_2d_utilities import fit_2dgaussian
import matplotlib.pyplot as plt
from astropy import coordinates as coords
from astropy import units as u
from astropy.time import Time
import angles as a
from astropy.coordinates import EarthLocation
from photutils import RectangularAperture,aperture_photometry

path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'sphere_data')

def sphere_transmission(BB_filter='B_H', DB_filter=None, NDset=0.):
    """
    
    Input:
        - BB_filter: name of the broad band filter (among 'B_H',
                    'B_Hcmp2', 'B_J', 'B_Ks', 'B_ND-H'). By default, assumes 'B_H'
        - DB_filter: name of the dual band filter. If None, assumes no DB filter
        - NDset: ND filter (float) among 0., 1., 2. or 3.5. By default 0.
    """
    # BB filter
    data_bb = ascii.read(os.path.join(path,'SPHERE_IRDIS_'+BB_filter+'.txt'))
    w_bb = data_bb['col1']
    t_bb = data_bb['col2']
    # DBI filter
    if (DB_filter != None and DB_filter != 'P0-90'):
        data_nb = ascii.read(os.path.join(path,'SPHERE_IRDIS_'+DB_filter+'.txt'))
        w_db = data_nb['col1']  
        t_db_1 = data_nb['col2']        
        t_db_2 = data_nb['col3']        
    else:
        w_db = w_bb
        t_db_1 = np.zeros(len(w_bb))+1.
        t_db_2 = np.zeros(len(w_bb))+1.
    # ND CPI
    data_nd = ascii.read(os.path.join(path,'SPHERE_CPI_ND.txt'))
    w_nd = data_nd['col1']
    if float(NDset) == 0.:
        t_nd = data_nd['col2']
    elif float(NDset) == 1.:
        t_nd = data_nd['col3']
    elif float(NDset) == 2.:
        t_nd = data_nd['col4']
    elif float(NDset) == 3.5:
        t_nd = data_nd['col5']
    else:
        print('I did not understand your choice of ND filter: {0:3.2f}'.format(NDset))
        return
    # interpolation
    lambdainterp  = np.arange(900,2401,1)

    interp_function_bb = interp1d(w_bb,t_bb)
    t_bb_interp = interp_function_bb(lambdainterp)
    t_bb_interp[t_bb_interp<0.]=0.

    interp_function_db_1 = interp1d(w_db,t_db_1)
    t_db_interp_1 = interp_function_db_1(lambdainterp)
    t_db_interp_1[t_db_interp_1<0.]=0.
    interp_function_db_2 = interp1d(w_db,t_db_2)
    t_db_interp_2 = interp_function_db_2(lambdainterp)
    t_db_interp_2[t_db_interp_2<0.]=0.

    interp_function_nd = interp1d(w_nd,t_nd)
    t_nd_interp = interp_function_nd(lambdainterp)
    t_nd_interp[t_nd_interp<0.]=0.

    t_final_1 = np.sum(t_bb_interp * t_db_interp_1)
    t_final_2 = np.sum(t_bb_interp * t_db_interp_2)
    t_final_nd_1 = np.sum(t_bb_interp * t_db_interp_1*t_nd_interp)
    t_final_nd_2 = np.sum(t_bb_interp * t_db_interp_2*t_nd_interp)
    
    return np.asarray([t_final_nd_1 / t_final_1,t_final_nd_2 / t_final_2])

def ifs_transmission(wavelength_array, NDset=0.,plot=False):
    """
    create a 39 element array -evenly spaced- starting from the minimum wavelength and 
    ending with max wavelength (parameters of this function)
    represent lambda for each channel
    Input:
        - wavelength_array: a numpy array containing the central wavelengths of the
            IFS channels in nm (typically an 39-element array).
        - NDset: a float representing the neutral density attenuation:
            can be either 0., 1., 2. or 3.5
    Output:
        an array with the mean transmission of the IFS in the given channels
    """
    # ND CPI
    data_nd = ascii.read(os.path.join(path,'SPHERE_CPI_ND.txt'))      
    w_nd = data_nd['col1']

    if float(NDset) == 0.:
        t_nd = data_nd['col2']
    elif float(NDset) == 1.:
        t_nd = data_nd['col3']
    elif float(NDset) == 2.:
        t_nd = data_nd['col4']
    elif float(NDset) == 3.5:
        t_nd = data_nd['col5']
    else:
        print('I did not understand your choice of ND filter: {0:3.2f}'.format(NDset))
        return

    interp_function_nd = interp1d(w_nd,t_nd)

    channel_width = wavelength_array-np.roll(wavelength_array,1)
    channel_width[0]=channel_width[1]
    
    transmission_array = np.zeros_like(wavelength_array)
    for i,wl in enumerate(wavelength_array):
        # we build an array representing the wavelengths of the channel, roughly spaced every nm
        channel_wl_array = np.linspace(wl-channel_width[i]/2,wl+channel_width[i]/2,num=int(channel_width[i]),endpoint=True)
        # we interpolate the ND transmission at those wavelengths        
        t_nd_interp = interp_function_nd(channel_wl_array)        
        t_nd_interp[t_nd_interp<0.]=0. # just in case
        transmission_array[i]=np.mean(t_nd_interp)
    if plot:
        plt.plot(w_nd,t_nd,color='black',label='ND{0:3.1f} tabulated transmission'.format(NDset))
        plt.plot(wavelength_array,transmission_array,'ro',label='Interpolation for the desired IFS channels')
        plt.xlabel('Wavelength in nm')        
        plt.ylabel('Transmission')
        plt.legend(frameon=False)
    return transmission_array

def zimpol_transmission(NDset=1.,filt='VBB'):
    """
    """
    # ND filter
    data_nd = ascii.read(os.path.join(path,'transmission_ND_filters.txt'))
    w_nd = data_nd['col1']
    if float(NDset) == 0.:
        t_nd = data_nd['col1']*0.
    elif float(NDset) == 1.:
        t_nd = data_nd['col2']
    elif float(NDset) == 2.:
        t_nd = data_nd['col3']
    elif float(NDset) == 4.:
        t_nd = data_nd['col4']
    else:
        print('I did not understand your choice of ND filter: {0:3.2f}'.format(NDset))
        return

    # Filter
    data_filt = ascii.read(os.path.join(path,'transmission_{0:s}.txt'.format(filt)))
    w_filt = data_filt['col1']
    t_filt = data_filt['col2']

    lambdainterp  = np.arange(400,1001,1)
    
    interp_function_nd = interp1d(w_nd,t_nd)
    t_nd_interp = interp_function_nd(lambdainterp)
    t_nd_interp[t_nd_interp<0.]=0.
    
    interp_function_filt = interp1d(w_filt,t_filt)
    t_filt_interp = interp_function_filt(lambdainterp)
    t_filt_interp[t_filt_interp<0.]=0.
    
    t_final = np.sum(t_filt_interp)  
    t_final_nd = np.sum(t_nd_interp * t_filt_interp)  
    
    return t_final_nd/t_final 

def waffle_spot_positions(centerxy,filter_name='B_H',waffle_pattern='x'):
    """
    Return a tuple with coordinates X,Y of the satellite spots in this order:
    upper left, upper right, lower left, lower right (ready to be used in 
    vip.calib.recentering.cube_recenter_satspots)
    Input:
        - filter_name among 'B_Y', 'B_J', 'B_H', 'B_Ks'
        - star_center: star approximate center position [x,y]
        - waffle_pattern: 'x' (default) or '+'
    Output:
        - list of 4 pairs [x,y] given the waffle spot center for the 
        upper left, upper right, lower left and lower right waffle.
    """
    if waffle_pattern == 'x':
        spot_pa = np.deg2rad(np.array([45.,135.,225.,315.]))
    else:
        print('Warning, function never tested for "+"...')
        spot_pa = np.deg2rad(np.array([0.,90.,180.,270.]))
    if filter_name == 'B_Y':
        radius = 32.
    elif filter_name == 'B_J':
        radius = 35.
    elif filter_name == 'B_H' or filter_name == 'B_ND-H':
        radius = 47.
    elif filter_name == 'B_Ks' or filter_name == 'D_K12':
        radius = 67.
    else:    
        print('Filter choice not understood:',filter_name)
    waffle_ur_pos = [int(np.fix(centerxy[0]+radius*np.cos(spot_pa[0]))),\
                     int(np.fix(centerxy[1]+radius*np.sin(spot_pa[0])))]
    waffle_ul_pos = [int(np.fix(centerxy[0]+radius*np.cos(spot_pa[1]))),\
                     int(np.fix(centerxy[1]+radius*np.sin(spot_pa[1])))]
    waffle_ll_pos = [int(np.fix(centerxy[0]+radius*np.cos(spot_pa[2]))),\
                     int(np.fix(centerxy[1]+radius*np.sin(spot_pa[2])))]
    waffle_lr_pos = [int(np.fix(centerxy[0]+radius*np.cos(spot_pa[3]))),\
                     int(np.fix(centerxy[1]+radius*np.sin(spot_pa[3])))]
    output = (waffle_ul_pos,waffle_ur_pos,waffle_ll_pos,waffle_lr_pos)
    return output

def waffle_spot_positions_ifs(centerxy,mode='YJ',waffle_pattern='x'):
    """
    Return a tuple with coordinates X,Y of the satellite spots in this order:
    upper left, upper right, lower left, lower right (ready to be used in 
    vip.calib.recentering.cube_recenter_satspots)
    Input:
        - filter_name among 'B_Y', 'B_J', 'B_H', 'B_Ks'
        - star_center: star approximate center position [x,y]
        - waffle_pattern: 'x' (default) or '+'
    Output:
        - list of 4 pairs [x,y] given the waffle spot center for the 
        upper left, upper right, lower left and lower right waffle.
    """
    if waffle_pattern == 'x':
        spot_pa = np.deg2rad(np.array([55.,145.,235.,325.]))
    else:
        print('Warning, function never tested for "+"...')
        spot_pa = np.deg2rad(np.array([10.,100.,190.,280.]))
    if mode == 'YJ':
        radius = np.linspace(48,67,39)
#    elif mode == 'YJH':
#        radius = 35.
    else:    
        print('Mode not understood')
    waffle_posxy = np.ndarray((4,39,2),dtype=int)
    for i_pa in range(4):
        waffle_posxy[i_pa,:,0] = np.asarray(np.round(centerxy[0]+radius*np.cos(spot_pa[i_pa])),dtype=int) # y center
        waffle_posxy[i_pa,:,1] = np.asarray(np.round(centerxy[1]+radius*np.sin(spot_pa[i_pa])),dtype=int) # x center
    return waffle_posxy

def theoretical_sphere_fwhm(filter_name='B_H',mirror=8.,verbose=True):
    """
    Returns the size of the theoretical FWHM in pixels,
    corresponding to the upper wavelength of the filter.
    Input:
        - filter_name: among 'B_Y', 'B_J', 'B_H', 'B_Ks'
        - mirror (optional): size of the mirror in m (by default 8m)
    Output:
        - array with the left and right FWHM (different in case of DBI filter)
    """
    if filter_name == 'B_Y' or filter_name == 'BB_Y':
        lambda_c=np.ones(2)*1042.5
        delta_lambda=np.ones(2)*139.
        pixel_scale=12.25/1000.
    elif filter_name == 'B_J' or filter_name == 'BB_J':
        lambda_c=np.ones(2)*1257.5
        delta_lambda=np.ones(2)*197.
        pixel_scale=12.25/1000.
    elif filter_name == 'B_H' or filter_name == 'B_ND-H' or filter_name == 'BB_H' or filter_name == 'DP_0_BB_H':
        lambda_c=np.ones(2)*1625.5
        delta_lambda=np.ones(2)*291.
        pixel_scale=12.25/1000.
    elif filter_name == 'B_Ks' or filter_name == 'BB_Ks':
        lambda_c=np.ones(2)*2181.3
        delta_lambda=np.ones(2)*313.5
        pixel_scale=12.25/1000.
    elif filter_name == 'DB_K12':
        lambda_c=np.array([2102.5,2255.])
        delta_lambda=np.array([102.,109.])
        pixel_scale=12.25/1000.
    elif filter_name == 'DB_H23':
        lambda_c=np.array([1588.8,1667.1])
        delta_lambda=np.array([53.1,55.6])        
        pixel_scale=12.25/1000.
    elif filter_name == 'YJ':
        lambda_c=np.linspace(960.,1340.,39)
        delta_lambda=np.median(lambda_c-np.roll(lambda_c,1))        
        pixel_scale=7.46/1000.
    elif filter_name == 'YJH':
        lambda_c=np.linspace(970.,1660.,39)
        delta_lambda=np.median(lambda_c-np.roll(lambda_c,1))        
        pixel_scale=7.46/1000.
    else:    
        print('Filter choice not understood')
    theoretical_fwhm = np.rad2deg((lambda_c+delta_lambda/2)*1e-9/mirror)*3600/pixel_scale
    if verbose:
        print('Theoretical FWHM: {0:4.2f} (left), {1:4.2f} (right)'.format(theoretical_fwhm[0],theoretical_fwhm[1]))
    return theoretical_fwhm

def sph_irdis_centerspot(waffle_cube, filter_name='B_H', centerguessxy=None,waffle_pattern='x',\
                        path='.',rspot=12,name='waffle',save=True,sigfactor=6):
    """
    Returns the position of the center of the frames and the shift to apply to recenter the images, as a tuple (center,shift).
    Each element is an array of dimension (nframes,2) with the y and x coordinates in the 2nd dimension and the frame number in the
    first dimentsion
    Input:
        - waffle_cube: the cube or frame with the waffle spots
        - filter_name: the name of the filter, as required from the function waffle_spot_positions
        - centerguessxy: the list [x0,y0]  where the approximate center is located
        - waffle_pattern: 'x' or '+'  as required from the function waffle_spot_positions
        - path: the path name where to store the results if save=True
        - rspot: the half side of the sub-images in which the 2d gaussian fit is made, by default 12
        - name: the name of the region file that will be saved
        - save: bool wether to save or not the ds9 region file (True by default)
        - sigfactor: the number of stdev for sigma thresholding before the detection 
        of the peak (6 by default)
    Output
        - centerfitxy: 2-element array with the center in X (first value) and Y (2nd value)
        - shift:  2-element array with the shift in X (first value) and Y (2nd value) to recenter the array
    """
    if len(waffle_cube.shape) == 2:
        nframes = 1
        ny,nx =  waffle_cube.shape
    else:
        nframes,ny,nx = waffle_cube.shape
    if centerguessxy is None:
        centerguessxy=[nx/2,ny/2]
    spot_pos = np.asarray(waffle_spot_positions(centerguessxy,filter_name=filter_name,waffle_pattern=waffle_pattern))
    subimage_names = ['upper_left','upper_right','lower_left','lower_right']
    spot_pos_fitted = np.ndarray((nframes,4,2))
    centerfitxy = np.ndarray((nframes,2))
    fwhm = theoretical_sphere_fwhm(filter_name)[0]
    for i in range(nframes):
        if len(waffle_cube.shape) == 2:
            img = waffle_cube
        else:
            img = waffle_cube[i,:,:]
        for j,subimage_name in enumerate(subimage_names):
            # upper left, upper right, lower left, lower right
            subimage = img[spot_pos[j,1]-rspot:spot_pos[j,1]+rspot,spot_pos[j,0]-rspot:spot_pos[j,0]+rspot]
            fig, ax = plt.subplots()
            ax.imshow(subimage, cmap='CMRmap', origin='lower',interpolation='nearest')
            ycen,xcen = fit_2dgaussian(subimage, crop=False, cent=None, cropsize=15, fwhmx=fwhm, \
                fwhmy=fwhm, sigfactor=sigfactor,threshold=True,verbose=True,plot=True)
            spot_pos_fitted[i,j,0] = spot_pos[j,0] - rspot + xcen[0] 
            spot_pos_fitted[i,j,1] = spot_pos[j,1] - rspot + ycen[0]
#            print('Center frame {0:02d} {1:s}: ({2:6.2f} {3:6.2f})'.format(i,subimage_name,\
#                spot_pos_fitted[i,j,0],spot_pos_fitted[i,j,1]))
#            if save:
#                fits.writeto(os.path.join(path,'frame_{0:02d}_{1:s}.fits'.format(i,subimage_name)),img,clobber=True)
        # line1 equation
#       a1 = (y_lr[1] - y_ul[1]) / (x_lr[0] - x_ul[0])				
        a1 = (spot_pos_fitted[i,3,1] - spot_pos_fitted[i,0,1]) / (spot_pos_fitted[i,3,0] - spot_pos_fitted[i,0,0])				
#        b1 = xy_lr[1]-a1*xy_lr[0]
        b1 = spot_pos_fitted[i,3,1]-a1*spot_pos_fitted[i,3,0]
        # line2 equation
#        a2 = (xy_ur[1] - xy_ll[1]) / (xy_ur[0] - xy_ll[0])				
        a2 = (spot_pos_fitted[i,1,1] - spot_pos_fitted[i,2,1]) / (spot_pos_fitted[i,1,0] - spot_pos_fitted[i,2,0])				
#        b2 = xy_ur[1]-a2*xy_ur[0]
        b2 = spot_pos_fitted[i,1,1]-a2*spot_pos_fitted[i,1,0]
        #intersection
        centerfitxy[i,0] = (b2-b1)/(a1-a2) 
        centerfitxy[i,1] = a1*centerfitxy[i,0]+b1	
        print('Center frame {0:02d}: ({1:6.2f} {2:6.2f})'.format(i,\
                centerfitxy[i,0] ,centerfitxy[i,1]))
        if save:            
            f = open(os.path.join(path,'{0:s}_frame_{1:02d}.reg'.format(name,i)),'w')
            f.write("global color=green dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=0 delete=1 include=1 source=1\n")
            f.write("physical\n")
            f.write('circle({0:8.2f},{1:8.2f},2.5)\n'.format(centerfitxy[i,0]+1,centerfitxy[i,1]+1))
            f.write('line({0:8.2f},{1:8.2f},{2:8.2f},{3:8.2f})\n'.format(spot_pos_fitted[i,3,0]+1,spot_pos_fitted[i,3,1]+1,spot_pos_fitted[i,0,0]+1,spot_pos_fitted[i,0,1]+1))
            f.write('line({0:8.2f},{1:8.2f},{2:8.2f},{3:8.2f})\n'.format(spot_pos_fitted[i,2,0]+1,spot_pos_fitted[i,2,1]+1,spot_pos_fitted[i,1,0]+1,spot_pos_fitted[i,1,1]+1))
            f.close() # you can omit in most cases as the destructor will call it            
        shift = centerfitxy - np.ones(((nframes,2)))*(nx/2)
    return centerfitxy,shift

def sph_ifs_centerspot(waffle_cube, mode='YJ', centerguessxy=None,waffle_pattern='x',\
                        path='.',rspot=12,name='waffle',save=True,sigfactor=6):
    """
    Returns the position of the center of the frames and the shift to apply to recenter the images, as a tuple (center,shift).
    Each element is an array of dimension (nframes,2) with the y and x coordinates in the 2nd dimension and the frame number in the
    first dimentsion
    Input:
        - waffle_cube: the cube or frame with the waffle spots
        - mode: the name of the filter, as required from the function waffle_spot_positions
        - centerguessxy: the list [x0,y0]  where the approximate center is located
        - waffle_pattern: 'x' or '+'  as required from the function waffle_spot_positions
        - path: the path name where to store the results if save=True
        - rspot: the half side of the sub-images in which the 2d gaussian fit is made, by default 12
        - name: the name of the region file that will be saved
        - save: bool wether to save or not the ds9 region file (True by default)
        - sigfactor: the number of stdev for sigma thresholding before the detection 
        of the peak (6 by default)
    Output
        - centerfitxy: 2-element array with the center in X (first value) and Y (2nd value)
        - shift:  2-element array with the shift in X (first value) and Y (2nd value) to recenter the array
    """
    if len(waffle_cube.shape) == 3:
        nframes = 1
        nlambda,ny,nx =  waffle_cube.shape
    else:
        nframes,nlambda,ny,nx = waffle_cube.shape
    if centerguessxy is None:
        centerguessxy=[nx/2,ny/2]
    spot_pos = waffle_spot_positions_ifs(centerguessxy,mode=mode,waffle_pattern=waffle_pattern)
    subimage_names = ['upper_left','upper_right','lower_left','lower_right']
    subimages_indices = [1,0,2,3]
    spot_pos_fitted = np.ndarray((nframes,4,nlambda,2))
    centerfitxy = np.ndarray((nframes,nlambda,2))
    fwhm_array = theoretical_sphere_fwhm(mode)
    for i in range(nframes):
        if len(waffle_cube.shape) == 3:
            spectral_cube = waffle_cube
        else:
            spectral_cube = waffle_cube[i,:,:,:]
        for ilambda in range(nlambda):
            for j,subimages_index in enumerate(subimages_indices):
                # upper left, upper right, lower left, lower right
                subimage = spectral_cube[ilambda,spot_pos[subimages_index,ilambda,1]-rspot:spot_pos[subimages_index,ilambda,1]+rspot,\
                                         spot_pos[subimages_index,ilambda,0]-rspot:spot_pos[subimages_index,ilambda,0]+rspot]
                fig, ax = plt.subplots()
                ax.imshow(subimage, cmap='CMRmap', origin='lower',interpolation='nearest')
                ycen,xcen = fit_2dgaussian(subimage, crop=False, cent=None, cropsize=15, fwhmx=fwhm_array[ilambda], \
                    fwhmy=fwhm_array[ilambda],sigfactor=sigfactor,threshold=True,verbose=True,plot=False)
                spot_pos_fitted[i,j,ilambda,0] = spot_pos[subimages_index,ilambda,0] - rspot + xcen[0] 
                spot_pos_fitted[i,j,ilambda,1] = spot_pos[subimages_index,ilambda,1] - rspot + ycen[0]
            a1 = (spot_pos_fitted[i,3,ilambda,1] - spot_pos_fitted[i,0,ilambda,1]) / (spot_pos_fitted[i,3,ilambda,0] - spot_pos_fitted[i,0,ilambda,0])				
            b1 = spot_pos_fitted[i,3,ilambda,1]-a1*spot_pos_fitted[i,3,ilambda,0]
            a2 = (spot_pos_fitted[i,1,ilambda,1] - spot_pos_fitted[i,2,ilambda,1]) / (spot_pos_fitted[i,1,ilambda,0] - spot_pos_fitted[i,2,ilambda,0])				
            b2 = spot_pos_fitted[i,1,ilambda,1]-a2*spot_pos_fitted[i,1,ilambda,0]
            centerfitxy[i,ilambda,0] = (b2-b1)/(a1-a2) 
            centerfitxy[i,ilambda,1] = a1*centerfitxy[i,ilambda,0]+b1	
            print('Center frame {0:02d} channel {1:02d}: ({2:6.2f} {3:6.2f})'.format(i,\
                    ilambda,centerfitxy[i,ilambda,0] ,centerfitxy[i,ilambda,1]))
            if save:            
                f = open(os.path.join(path,'{0:s}_frame_{1:02d}_channel{2:02d}.reg'.format(name,i,ilambda)),'w')
                f.write("global color=green dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=0 delete=1 include=1 source=1\n")
                f.write("physical\n")
                f.write('circle({0:8.2f},{1:8.2f},2.5)\n'.format(centerfitxy[i,ilambda,0]+1,centerfitxy[i,ilambda,1]+1))
                f.write('line({0:8.2f},{1:8.2f},{2:8.2f},{3:8.2f})\n'.format(spot_pos_fitted[i,3,ilambda,0]+1,spot_pos_fitted[i,3,ilambda,1]+1,spot_pos_fitted[i,0,ilambda,0]+1,spot_pos_fitted[i,0,ilambda,1]+1))
                f.write('line({0:8.2f},{1:8.2f},{2:8.2f},{3:8.2f})\n'.format(spot_pos_fitted[i,2,ilambda,0]+1,spot_pos_fitted[i,2,ilambda,1]+1,spot_pos_fitted[i,1,ilambda,0]+1,spot_pos_fitted[i,1,ilambda,1]+1))
                f.close() # you can omit in most cases as the destructor will call it            
            shift = centerfitxy - np.ones(((nframes,nlambda,2)))*(nx/2)
    return centerfitxy,shift

def get_companion_position(centerxy,sep,pa,parang,rotoff=0):
    """
    Given a pupil-stabilized cube with the star in centerxy=(x,y) with a sequence 
    of parallactic angle and rotoff, this functions computes the coordinates x and y
    of an object fixed in the sky with a separation and position angle  sep and pa
    with respect to the star. 
    """
    angle = parang-rotoff
    xcoords = centerxy[0] - sep*np.cos(np.deg2rad(pa+angle))
    ycoords = centerxy[1] + sep*np.sin(np.deg2rad(pa+angle))
    return xcoords,ycoords

def analyse_coords_from_header(filename,silent=False):
    """
    """
    if filename.endswith('.fits'):
        header = fits.getheader(filename)
    else:
        print('Function not yet implemented, sorry')
        return
    latitude = -24.6268 #header['HIERARCH ESO TEL GEOLAT']*u.degree
    longitude = -70.4045 #header['HIERARCH ESO TEL GEOLON']*u.degree  
    altitude = 2648.0 #header['HIERARCH ESO TEL GEOELEV']*u.meter 
#    pma = header['HIERARCH ESO TEL TARG PMA']
#    pmd = header['HIERARCH ESO TEL TARG PMD']   
    date = Time(header['DATE-OBS'],location=(longitude, latitude, altitude))
#    date = Time(header['DATE'],location=(longitude, latitude, altitude))
    fk5_timeOfObservation = coords.FK5(equinox=date.jyear_str)
    print(' ')
#    # Simbad 
#    try:
#        star = header['HIERARCH ESO OBS TARG NAME']
#        J2000_simbad_coords = coords.SkyCoord.from_name(star)
#        print('ICRS J2000.000 from Simbad {0:s}'.format(J2000_simbad_coords.to_string('hmsdms')))
#    except:
#        print("Error:", sys.exc_info()[0])
#    # OB
#    J2000_target_alpha_ob = a.convert_keyword_coord(header['HIERARCH ESO TEL TARG ALPHA'])
#    J2000_target_delta_ob = a.convert_keyword_coord(header['HIERARCH ESO TEL TARG DELTA'])
#    J2000_target_coords = coords.SkyCoord(J2000_target_alpha_ob,J2000_target_delta_ob,\
#        frame='icrs',unit=(u.hourangle, u.deg))
#    print('ICRS J2000.000 TEL.TARG.ALPHA/DELTA {0:s}'.format(J2000_target_coords.to_string('hmsdms')))       
#    Jcurrent_coords_target_wo_ppm_correction = J2000_target_coords.transform_to(fk5_timeOfObservation)
#    Jcurrent_coords_target_w_ppm_correction = coords.SkyCoord( \
#            Jcurrent_coords_target_wo_ppm_correction.ra +(date.jyear-2000)*pma*u.arcsec/np.cos(Jcurrent_coords_target_wo_ppm_correction.dec),\
#            Jcurrent_coords_target_wo_ppm_correction.dec+(date.jyear-2000)*pmd*u.arcsec, 
#            frame=fk5_timeOfObservation)
#    print('FK5 J{0:8.3f} TEL.TARG.ALPHA/DELTA {1:s}'.format(date.jyear,\
#        Jcurrent_coords_target_w_ppm_correction.to_string('hmsdms')))
        
    # Pointing
    pointing_alpha = header['RA']*u.degree
    pointing_delta = header['DEC']*u.degree
    J2000_pointing_coords = coords.SkyCoord(pointing_alpha,pointing_delta,frame='fk5')
    print('FK5  J2000.000 pointing (RA/DEC) {0:s}'.format(J2000_pointing_coords.to_string('hmsdms')))
    Jcurrent_pointing_coords = J2000_pointing_coords.transform_to(fk5_timeOfObservation)
    print('FK5 J{0:8.3f} pointing (RA/DEC) {1:s}'.format(date.jyear,Jcurrent_pointing_coords.to_string('hmsdms')))
    
    # derotator
    drot_alpha = a.convert_keyword_coord(header['HIERARCH ESO INS4 DROT2 RA'])
    drot_delta = a.convert_keyword_coord(header['HIERARCH ESO INS4 DROT2 DEC'])
    Jcurrent_drot_coords = coords.SkyCoord(drot_alpha,drot_delta, frame=fk5_timeOfObservation,unit=(u.hourangle, u.deg))
    print('FK5  J{0:8.3f} INS4.DROT2.RA/DEC {1:s}'.format(date.jyear, Jcurrent_drot_coords.to_string('hmsdms')))
#    print('Proper motion from the header: pma={0:6.4f}"/yr, pmd={1:6.4f}"/yr'.format(pma,pmd))

    
    Jcurrent_pointing_coords.location = EarthLocation.of_site(u'Paranal Observatory')
    Jcurrent_pointing_coords.obstime=date
    
    Jcurrent_drot_coords.location = EarthLocation.of_site(u'Paranal Observatory')
    Jcurrent_drot_coords.obstime=date
    
    diff_alt = Jcurrent_pointing_coords.altaz.alt-Jcurrent_drot_coords.altaz.alt
    diff_az = Jcurrent_pointing_coords.altaz.az-Jcurrent_drot_coords.altaz.az

    print('Diff : {0:.1f} (alt) {1:.1f} (az) '.format(\
          diff_alt.to(u.arcsec).value,diff_az.to(u.arcsec).value))

#sh = sin(ha*d2r) & ch = cos(ha*d2r)
#sd = sin(dec*d2r) & cd = cos(dec*d2r)
#sl = sin(lat*d2r) & cl = cos(lat*d2r)

#    sh = np.sin(Jcurrent_coords_target_w_ppm_correction.)
#    return diff_alt.to(u.arcsec).value,diff_az.to(u.arcsec).value
    return Jcurrent_pointing_coords,Jcurrent_drot_coords

def spider_aperture_photometry(cube,rin,rout,width=4,full_output=False):
    """
    Function that performs apterure photometry in rectangle oriented along the direction
    of the spiders and starting at a radius rin up to a radius rout with a width of 4 pixels by default
    Input:
        - cube: a cube or a frame
        - rin: the inner radius to start the aperture photmetry
        - rout: the outer radius to stop the aperture photometry (the rectangle
        has a length of rout-rin)
        - width: the size of the rectangle perpendicular to the spider direction
        - full_output: if yes returns a multimensional array (nframes,4) with the
        aperture photometry for each spider. Else returns only the mean value for each frame
    Output:
        see full_output
    """
    shape=cube.shape
    if len(shape)==2:
        nframes=1
        sizex=shape[1]
        sizey=shape[0]
        cube = cube.reshape(1,sizey,sizex)
    else:
        nframes=shape[0]
        sizey=shape[1]
        sizex=shape[2]
    spider_pa = np.array([40,140,220,320])
    spider_pa_sky = spider_pa+20
    center_radius = (rin+rout)/2.
    xcenter = sizex//2+center_radius*np.cos(np.deg2rad(spider_pa))
    ycenter = sizey//2+center_radius*np.sin(np.deg2rad(spider_pa))
    aper_phot = np.ndarray((nframes,4))
    aper_phot_sky = np.ndarray((nframes,4))
    for i in range(4):
        aper = RectangularAperture((xcenter[i],ycenter[i]), rout-rin, width, np.deg2rad(spider_pa[i]))
        aper_sky = RectangularAperture((xcenter[i],ycenter[i]), rout-rin, width, np.deg2rad(spider_pa_sky[i]))
        for k in range(nframes):
            aper_phot[k,i] = aperture_photometry(cube[k,:,:],aper)['aperture_sum'][0]
            aper_phot_sky[k,i] = aperture_photometry(cube[k,:,:],aper_sky)['aperture_sum'][0]
    if full_output:
        return aper_phot - aper_phot_sky
    else:
        return np.mean(aper_phot,axis=1) - np.mean(aper_phot_sky,axis=1)
    
if __name__ == "__main__":
#    tr = sphere_transmission(BB_filter='B_H', DB_filter=None, NDset=1.)
    tr = sphere_transmission(BB_filter='B_Ks', DB_filter=None, NDset=2.)
    print(tr)
#    theoretical_sphere_fwhm(filter_name='YJ')
#    tr = zimpol_transmission(NDset=1.,filt='VBB')
#    print(tr)

#    ap = spider_aperture_photometry(cube_rebin,60,80,width=8,full_output=True)
#    plt.plot(ap,color='blue')
##    plt.plot(ap[:,1],color='red')
#    plt.grid()
#    threshold = np.arange(10,30,2)
#    fraction_filtered = np.asarray([np.sum(ap>thresh)*100./len(ap) for thresh in threshold])
#    plt.plot(threshold,fraction_filtered)   
#    bad_frames = ap>10
#    good_frames = ap<=10
