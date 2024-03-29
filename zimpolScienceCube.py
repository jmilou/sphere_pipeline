    # -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:22:00 2015
@author: jmilli
Modified on 2017-04-16 to add the keyword save to the method computeStokes
Modified on 2017-04-16 to save the mean beamshift 
"""
from zimpolMasterFile import ZimpolMasterFile
from zimpolMasterBias import ZimpolMasterBias
from zimpolMasterFlat import ZimpolMasterFlat
from cosmetics_julien import correctZimpolBadPixelInCube
from astropy.io import fits,ascii
import os
import glob
import numpy as np
#from image_tools import shift_cube
import image_tools as imtools
from shiftFinder import ShiftFinder
#from vip.var import fit_2dgaussian # for the recentering function
#from vip.calib import frame_shift # for the recentering function
#from vip.var import frame_filter_gaussian2d # for the recentering function


class ZimpolScienceCube(ZimpolMasterFile):
    """
    This is a master science frame for Zimpol. It represents a set of files taken
    with the same position of the HWP corresponding to a Stokes parameter (+Q,
    -Q, +U or -U).
    It inherits from the ZimpolMasterFile object.
    """
    
    def __init__(self,pathRaw,pathReduc,fileName,badPixelMap=None,\
                 name='zimpol_master_science_frame',masterBias=None,masterFlat=None,\
                 badPixel=False,guess_xycenter=None,biasOnlyI=True,recenter='default',\
                 fft=True):
        """
        Constructor of the class ZimpolScienceCube. It inherits from ZimpolMasterFile
        but takes an additional input for the masterDark to be subtracted and for 
        the masterFlat
    Common attributes with ZimpolMasterFile:
        - _pathRaw: the absolute path where the raw files are stored
        - _pathReduc: the absolute path where the reduced files
                      are stored
        - _fileName: the filename. It can be either a string 
                      with the start of the 
                      file name, e.g. 'SPHERE_ZIMPOL_', or the complete filename
        - _keywords: a dictionary on keywords. Each element of the dictionnary  
                     is a list of the keyword values of each file.
        - _prescanColumns
        - _overscanColumns
        - _columnNb
        - _rowNb
        - _keywordList
        - _cameras
        - _frameTypes
        - _name
        - _masterFrame_cam1 : the master frame image (initially nan)
        - _rmsMap_cam1 : the rms of the cube along the z axis (initially nan)
        - _weightMap_cam1 : the number of z values used to build the master frame
                  (initially 0)
        - _badPixelMap_cam1 : the map of bad pixels (initially 0)
        - _masterFrame_cam2 : the master frame image (initially nan)
        - _rmsMap_cam2 : the rms of the cube along the z axis (initially nan)
        - _weightMap_cam2 : the number of z values used to build the master frame
                  (initially 0)
        - _badPixelMap_cam2 : the map of bad pixels (initially 0)
        - _frames: a list of 2 dictionaries for cam1 and cam2 over the _frameTypes with the individual frames 
        Each of those attributes is a dictionnary containing the following entries:
        'pi_odd', 'pi_even', '0_even', and '0_odd'.
    Specific attributes to ZimpolScienceCube:
        - _centerx : the X centers of images (a list of 2 dictionaries for cam1 and cam2 over the _frameTypes)
        - _centery : the Y centers of images (a list of 2 dictionaries for cam1 and cam2 over the _frameTypes)        
        - _masterBias : an object ZimpolMasterBias to be subtracted to the flat.
        - _masterFlat : an object ZimpolMasterFlat to do the flat division.
        - _header :the header of the file
        - _mean_beamshift: the mean beam shift of cam1, cam2 in x and y
        - _std_beamshift: the dispersion in beam shift of cam1, cam2 in x and y
        - _guess_xy: a guess for the center of the cam1 and cam2 frames (array 
         of format [[cen_X_cam1,cen_Y_cam1],[cen_X_cam2,cen_Y_cam2]])
        - _biasOnlyI : boolean to specify whether the bias is only applied to I or to Q and U
    Common methods with ZimpolMasterFile:
        - writeMetaData
        - loadFiles
        - testPath
        - getNumberFiles
        - getFileNames
        - getKeywords
        - _extractFramesFromFile
        - extractFramesFromFiles
        - _extractFramesFromCube
        - rebinColumns
        - getRowNumber
        - getColumnNumber
        - getCameras
        - getFrameTypes
        - getName
        - getTotalNumberFrames
        - write
    Specific methods to ZimpolScienceCube:
        - buildMasterFrame: overloaded operator to be able to take a flat, bias 
        or modem into account.
        """
        ZimpolMasterFile.__init__(self,pathRaw,pathReduc,fileName,\
                                  badPixelMap,name,subtractDark=False)
#        if self.getNumberFiles() > 1:
#            raise Exception('The object ZimpolScienceCube accepts only one file')
        self._masterBias=masterBias
        self._masterFlat=masterFlat    
        self._cube_cam1  = self._frames[self.getCameras()[0]]
        self._cube_cam2 = self._frames[self.getCameras()[1]]
        self._header = self._frames['header']
        self.recenter = recenter
        if fft:
            self._shift = imtools.shift_cube
        else:
            self._shift = imtools.shift_cube_nofft
        self._fft = fft
        self._even_minus_odd = None
        self._even_plus_odd = None
        self._mean_beamshift = np.zeros((2,2,2)) #the different dimensions are cam phase X/Y
        self._std_beamshift = np.zeros((2,2,2)) #the different dimensions are cam phase X/Y
        if np.any(guess_xycenter) == None:
            self._guess_xy = np.asarray([[self._columnNb//2,self._rowNb//2],[self._columnNb//2,self._rowNb//2]])
        else:
            self._guess_xy = guess_xycenter
        self._biasOnlyI = biasOnlyI
        if masterBias!= None:
            if biasOnlyI==False:
                self.subtractBias()
        self.divideByFlat()
        if badPixel:
            self.correctBadPixels()

    def subtractBias(self):
        """
        Subtract the bias frame from individual images of the cube
        """
        if self._masterBias != None:
            for frameType in self.getFrameTypes():
                self._frames[self.getCameras()[0]][frameType] -= self._masterBias._masterFrame_cam1[frameType]       
                self._frames[self.getCameras()[1]][frameType] -= self._masterBias._masterFrame_cam2[frameType]       
        else:
            print('No master bias was provided')

    def divideByFlat(self):
        """
        Divide the individual images by the master flat
        """
        if self._masterFlat != None:
            for frameType in self.getFrameTypes():
                self._frames[self.getCameras()[0]][frameType] /= self._masterFlat._masterFrame_cam1[frameType]
                self._frames[self.getCameras()[1]][frameType] /= self._masterFlat._masterFrame_cam2[frameType]
        else:
            print('No master flat field was provided')

    def correctBadPixels(self,verbose=True):
        """
        Correct the bad pixels using the procedure
        correctZimpolBadPixelInCube(cube, cx,cy,size=5,radius=60,threshold=20,verbose=True)
        """
        for frameType in self.getFrameTypes():
            #cam 1:
            self._frames[self.getCameras()[0]][frameType] = correctZimpolBadPixelInCube(\
                         self._frames[self.getCameras()[0]][frameType],self._guess_xy[0,0],\
                         self._guess_xy[0,1],verbose=verbose,threshold=10,radius=20)
            #cam 2
            self._frames[self.getCameras()[1]][frameType] = correctZimpolBadPixelInCube(\
                         self._frames[self.getCameras()[1]][frameType],self._guess_xy[1,0],\
                         self._guess_xy[1,1],verbose=verbose,threshold=10,radius=20)                
                     
    def findCenter(self,method='default'):
        """
        Recenter the frames (before collapsing them)
        Input:
            - method: 'default', 'gaussianFit','saturated', or the name of a folder 
                containing the beamshift_correction_cam[1-2].csv
         """
        self._centerx = []
        self._centery = []
        for icam,cam in enumerate(self._cameras):
            dico_center_x = {} # dictionary with self._frameTypes=['0_even', '0_odd','pi_even','pi_odd'] as keys and a list of x centers  as items
            dico_center_y = {} # dictionary with self._frameTypes=['0_even', '0_odd','pi_even','pi_odd'] as keys and a list of x centers as items
            nb_frames = self._frames[cam][self._frameTypes[0]].shape[0]
            if method == 'default': # We just set the center to the default [size_y/2,size_x/2]
                for frameType in self._frameTypes:
                    dico_center_x[frameType] = np.zeros(nb_frames)+self._columnNb//2
                    dico_center_y[frameType] = np.zeros(nb_frames)+self._rowNb//2                
            elif method=='gaussianFit':
                dico = self._frames[cam]
                for frameType in self._frameTypes:
                    dico_center_x[frameType] = np.zeros(nb_frames)+self._columnNb//2
                    dico_center_y[frameType] = np.zeros(nb_frames)+self._rowNb//2     
                    for frame in range(nb_frames):
                        img = dico[frameType][frame,:,:]
                        mask_sky = np.zeros_like(img,dtype=bool)
                        mask_sky[:,0:200]=True
                        mask_sky[:,823:] =True
                        mask_fit = np.ones_like(img,dtype=bool)
                        mask_fit[:,0:200]=False
                        mask_fit[:,823:] =False
#                        # CONVOLUTION GAUSSIAN KERNEL to detect the star position
#                        conv_img = frame_filter_gaussian2d(img, 20)
#                        argmax = np.argmax(conv_img)
#                        ymax,xmax = np.unravel_index(argmax,conv_img.shape)
#                        print('Frame {0:d} type {1:s} Size {2:d}x{3:d} Guessed center X={4:d} , Y={5:d}'.format(frame,\
#                              frameType,conv_img.shape[1],conv_img.shape[0],xmax,ymax))
                        print('Frame {0:d} type {1:s} Size {2:d}x{3:d}'.format(frame,\
                              frameType,img.shape[1],img.shape[0]))

                        gauss2d_finder = ShiftFinder(img,crop=20,guess_xy=self._guess_xy[icam,:],sky=mask_sky,mask=mask_fit,threshold=120000.)#64000
                        name = os.path.join(self._pathReduc,'{0:s}_frame{1:03d}_{2:s}_cam{3:d}'.format(self._name,frame,frameType,cam))
                        print(os.path.basename(name))
                        fit_result,fit_error,chi2,chi2_reduced = gauss2d_finder.fit_gaussian(plot=True,\
                            save=name,verbose=False,fwhmx=15,fwhmy=5,theta=90)

                        dico_center_x[frameType][frame] = fit_result['X']
                        dico_center_y[frameType][frame] = fit_result['Y']
            elif method=='saturated':
                dico = self._frames[cam]
                for frameType in self._frameTypes:
                    dico_center_x[frameType] = np.zeros(nb_frames)+self._columnNb//2
                    dico_center_y[frameType] = np.zeros(nb_frames)+self._rowNb//2     
                    for frame in range(nb_frames):
                        img = dico[frameType][frame,:,:]
                        x_center_1d_array = np.arange(0,img.shape[1])
                        y_center_1d_array= np.arange(0,img.shape[0])
                        xx_array,yy_array=np.meshgrid(x_center_1d_array,y_center_1d_array)
                        mask_saturated = img>55000
                        for y in range(img.shape[0]):
                            if np.any(mask_saturated[y,:]):
                                minx = np.min(x_center_1d_array[mask_saturated[y,:]])
                                maxx = np.max(x_center_1d_array[mask_saturated[y,:]])                                
                                mask_saturated[y,minx:maxx+1] = True
                        center_x = np.mean(xx_array[mask_saturated])
                        center_y = np.mean(yy_array[mask_saturated])                   
                        x_center_1d_array_centered = (np.arange(0,img.shape[1])-center_x)*0.5
                        y_center_1d_array_centered = np.arange(0,img.shape[0])-center_y
                        xx_array_centered,yy_array_centered=np.meshgrid(x_center_1d_array_centered,y_center_1d_array_centered)
                        dist_cent = np.abs(xx_array_centered+1j*yy_array_centered)
                        max_saturated_radius = np.max(dist_cent[mask_saturated])
                        nb_px_saturated_mask = np.sum(mask_saturated*1)
                        print('The CG of saturated pixels is ({0:4.1f},{1:4.1f}) with an extension of {2:4.0f}px and contains {3:d}pixels'.format(center_x,center_y,max_saturated_radius,nb_px_saturated_mask))
                        dico_center_x[frameType][frame] = center_x
                        dico_center_y[frameType][frame] = center_y            
                        print('We use this CG as center for frames of type {0:s}'.format(frameType))                        
            else:
                # we assume here that the method points to a folder containing the list of the centers to be used in the same format
                center_filename = os.path.join(method,'{0:s}_cam{1:d}_center.txt'.format(self._name,cam)) # like folder/cycle00_HWP0000_cam1_center.txt
                center_files = glob.glob(os.path.join(method,'cycle*_HWP*_cam{0:d}_center.txt'.format(cam)))
                for frameType in self._frameTypes:
                    center_list_x = []
                    center_list_y = []
                    for center_filename in center_files:
                        print('Reading the center file {0:s}'.format(center_filename))
                        ascii_file = ascii.read(center_filename)
                        center_list_x = np.append(center_list_x,ascii_file[frameType+'_X'])
                        center_list_y = np.append(center_list_y,ascii_file[frameType+'_Y'])
                    mean_x_center = np.median(center_list_x)
                    mean_y_center = np.median(center_list_y)
                    distance_to_median_x = center_list_x - mean_x_center
                    distance_to_median_y = center_list_y - mean_x_center
                    mean_x_center = np.median(center_list_x[distance_to_median_x<10])
                    mean_y_center = np.median(center_list_y[distance_to_median_y<10])
                    dico_center_x[frameType] = np.ones(nb_frames)*mean_x_center
                    dico_center_y[frameType] = np.ones(nb_frames)*mean_y_center
                    print('Frames of type {0:s} will be centered at {1:.2f} in x and {2:.2f} in y'.format(frameType,\
                          mean_x_center,mean_y_center))

            if method != 'default' and method != 'gaussianFit' and method != 'saturated':
                # self._centerx and self._centery are already well defined. nb_frames is in memory
                dico = self._frames[cam]
                for izeroPi,zeroPi in enumerate(['0','pi']):
                    dico = self._frames[cam]
                    even_name = '{0:s}_even'.format(zeroPi)
                    odd_name = '{0:s}_odd'.format(zeroPi)
                    shift_array_1d = np.arange(0.1,1,0.1)-0.5
                    xshift_2d = np.ndarray((len(shift_array_1d),len(shift_array_1d)),dtype=float)
                    yshift_2d = np.ndarray((len(shift_array_1d),len(shift_array_1d)),dtype=float)
                    beamshift_optimization_matrix_median = np.ndarray((nb_frames,len(shift_array_1d),len(shift_array_1d)))
                    beamshift_optimization_matrix_std = np.ndarray((nb_frames,len(shift_array_1d),len(shift_array_1d)))
                    for frame in range(nb_frames):
                        img_even = dico[even_name][frame,:,:]
                        img_odd = dico[odd_name][frame,:,:]
                        x_center_1d_array = np.arange(0,img_even.shape[1])
                        y_center_1d_array= np.arange(0,img_even.shape[0])
                        xx_array,yy_array=np.meshgrid(x_center_1d_array,y_center_1d_array)
                        mask_saturated = np.logical_or(img_even>55000,img_odd>55000)
                        for y in range(img_even.shape[0]):
                            if np.any(mask_saturated[y,:]):
                                minx = np.min(x_center_1d_array[mask_saturated[y,:]])
                                maxx = np.max(x_center_1d_array[mask_saturated[y,:]])                                
                                mask_saturated[y,minx:maxx+1] = True
#                        fits.writeto(os.path.join(self._pathReduc,self._name+'_beamshift_saturation_mask_{0:s}_frame{1:02d}_cam{2:d}.fits'.format(zeroPi,frame,cam)),mask_saturated*1.,overwrite=True)
                        center_x = np.mean(xx_array[mask_saturated])
                        center_y = np.mean(yy_array[mask_saturated])                   
                        x_center_1d_array_centered = (np.arange(0,img_even.shape[1])-center_x)*0.5
                        y_center_1d_array_centered = np.arange(0,img_even.shape[0])-center_y
                        xx_array_centered,yy_array_centered=np.meshgrid(x_center_1d_array_centered,y_center_1d_array_centered)
                        dist_cent = np.abs(xx_array_centered+1j*yy_array_centered)
                        max_saturated_radius = np.max(dist_cent[mask_saturated])
                        nb_px_saturated_mask = np.sum(mask_saturated*1)
                        print('The CG of saturated pixels is ({0:4.1f},{1:4.1f}) with an extension of {2:4.0f}px and contains {3:d}pixels'.format(center_x,center_y,max_saturated_radius,nb_px_saturated_mask))
                        if max_saturated_radius>40:
                            print('Warning: saturation was detected beyond 40px, something is probably weird.')
                            print('We set the saturation radius to 10')
                            max_saturated_radius = 10
                        mask_for_fit = np.logical_and(dist_cent<max_saturated_radius*4,~mask_saturated)
                        diff = img_even-img_odd
                        diff[~mask_for_fit] = np.nan
                        med_ref = np.nanmedian(diff)
                        std_ref = np.nanstd(diff)
                        min_x_for_crop = np.min(xx_array[np.isfinite(diff)])
                        max_x_for_crop = np.max(xx_array[np.isfinite(diff)])
                        min_y_for_crop = np.min(yy_array[np.isfinite(diff)])
                        max_y_for_crop = np.max(yy_array[np.isfinite(diff)])
                        img_even_cropped = img_even[min_y_for_crop:max_y_for_crop+1,min_x_for_crop:max_x_for_crop+1]
                        img_odd_cropped = img_odd[min_y_for_crop:max_y_for_crop+1,min_x_for_crop:max_x_for_crop+1]
                        mask_for_fit_cropped = mask_for_fit[min_y_for_crop:max_y_for_crop+1,min_x_for_crop:max_x_for_crop+1]

                        median_shift = np.ndarray((len(shift_array_1d),len(shift_array_1d)),dtype=float)*np.nan
                        std_shift = np.ndarray((len(shift_array_1d),len(shift_array_1d)),dtype=float)*np.nan
                        rel_change_std_shift = np.ndarray((len(shift_array_1d),len(shift_array_1d)),dtype=float)*np.nan
                        rel_change_median_shift = np.ndarray((len(shift_array_1d),len(shift_array_1d)),dtype=float)*np.nan
                        for ixshift,xshift in enumerate(shift_array_1d):
                            for iyshift,yshift in enumerate(shift_array_1d):
                                xshift_2d[iyshift,ixshift] = xshift 
                                yshift_2d[iyshift,ixshift] = yshift 
                                img_shifted = imtools.shift_image_nofft(img_even_cropped,-xshift,-yshift,verbose=True)
                                diff = img_shifted-img_odd_cropped
                                diff[~mask_for_fit_cropped] = np.nan
                                median_shift[iyshift,ixshift] = np.nanmedian(diff)
                                std_shift[iyshift,ixshift] = np.nanstd(diff)
                                rel_change_std_shift[iyshift,ixshift] = (std_shift[iyshift,ixshift]-std_ref)/std_ref
                                rel_change_median_shift[iyshift,ixshift] = (median_shift[iyshift,ixshift]-med_ref)/med_ref
                        beamshift_optimization_matrix_median[frame,:,:] = median_shift
                        beamshift_optimization_matrix_std[frame,:,:] = std_shift
                        argmin = np.argmin(std_shift)
                        iyshift,ixshift=np.unravel_index(argmin,std_shift.shape)
                        xshift_best = xshift_2d[iyshift,ixshift]
                        yshift_best = yshift_2d[iyshift,ixshift]
                        print('Best beamshift: (X,Y)=({0:4.1f},{1:4.1f}) with a rel change in std of {2:4.1f}% and in median flux of {3:4.1f}%'.format(\
                              xshift_best,yshift_best,rel_change_std_shift[iyshift,ixshift]*100,rel_change_median_shift[iyshift,ixshift]*100))                        
                        dico_center_x[self._frameTypes[izeroPi*2]][frame] = dico_center_x[self._frameTypes[izeroPi*2+1]][frame]+xshift_best  # self._frameTypes[izeroPi*2] is 0 for 0 and 2 for pi
                        dico_center_y[self._frameTypes[izeroPi*2]][frame] = dico_center_y[self._frameTypes[izeroPi*2+1]][frame]+yshift_best  
                    fits.writeto(os.path.join(self._pathReduc,self._name+'_beamshift_matrix_median_{0:s}_cam{1:d}.fits'.format(zeroPi,cam)),beamshift_optimization_matrix_median,overwrite=True)
                    fits.writeto(os.path.join(self._pathReduc,self._name+'_beamshift_matrix_std_{0:s}_cam{1:d}.fits'.format(zeroPi,cam)),beamshift_optimization_matrix_std,overwrite=True)
                
            med_0_even_x  = np.median(dico_center_x[self._frameTypes[0]])
            med_0_even_y  = np.median(dico_center_y[self._frameTypes[0]])
            med_0_odd_x   = np.median(dico_center_x[self._frameTypes[1]])
            med_0_odd_y   = np.median(dico_center_y[self._frameTypes[1]])
            med_pi_even_x = np.median(dico_center_x[self._frameTypes[2]])
            med_pi_even_y = np.median(dico_center_y[self._frameTypes[2]])
            med_pi_odd_x  = np.median(dico_center_x[self._frameTypes[3]])
            med_pi_odd_y  = np.median(dico_center_y[self._frameTypes[3]])

            std_0_even_x  = np.std(dico_center_x[self._frameTypes[0]])
            std_0_even_y  = np.std(dico_center_y[self._frameTypes[0]])
            std_0_odd_x   = np.std(dico_center_x[self._frameTypes[1]])
            std_0_odd_y   = np.std(dico_center_y[self._frameTypes[1]])
            std_pi_even_x = np.std(dico_center_x[self._frameTypes[2]])
            std_pi_even_y = np.std(dico_center_y[self._frameTypes[2]])
            std_pi_odd_x  = np.std(dico_center_x[self._frameTypes[3]])
            std_pi_odd_y  = np.std(dico_center_y[self._frameTypes[3]])

            std_0_even_minus_odd_x = np.std(dico_center_x[self._frameTypes[0]]-dico_center_x[self._frameTypes[1]])
            std_0_even_minus_odd_y = np.std(dico_center_y[self._frameTypes[0]]-dico_center_y[self._frameTypes[1]])
            std_pi_even_minus_odd_x = np.std(dico_center_x[self._frameTypes[2]]-dico_center_x[self._frameTypes[3]])
            std_pi_even_minus_odd_y = np.std(dico_center_y[self._frameTypes[2]]-dico_center_y[self._frameTypes[3]])

            ascii.write([dico_center_x[self._frameTypes[0]],dico_center_x[self._frameTypes[1]],dico_center_x[self._frameTypes[2]],\
                         dico_center_x[self._frameTypes[3]],dico_center_y[self._frameTypes[0]],dico_center_y[self._frameTypes[1]],\
                         dico_center_y[self._frameTypes[2]],dico_center_y[self._frameTypes[3]]],\
                    os.path.join(self._pathReduc,'{0:s}_cam{1:d}_center.txt'.format(self._name,cam)),\
                    names=['0_even_X','0_odd_X','pi_even_X','pi_odd_X','0_even_Y','0_odd_Y','pi_even_Y','pi_odd_Y'])

            print('Summary {0:s} camera {1:d}'.format(self._name,cam))
            print('Intrinsic variations due to star jittering:')
            print('in 0 even : sig_X={0:3.2f} sig_Y={1:3.2f}'.format(std_0_even_x,std_0_even_y))
            print('in 0 odd  : sig_X={0:3.2f} sig_Y={1:3.2f}'.format(std_0_odd_x,std_0_odd_y))
            print('in pi even: sig_X={0:3.2f} sig_Y={1:3.2f}'.format(std_pi_even_x,std_pi_even_y))
            print('in pi odd : sig_X={0:3.2f} sig_Y={1:3.2f}'.format(std_pi_odd_x,std_pi_odd_y))
            print('Difference between even and odd frames (beamshift effect)')
            print('in 0  frames (median): delta_X={0:3.2f} delta_Y={1:3.2f}'.format(med_0_even_x-med_0_odd_x,med_0_even_y-med_0_odd_y))
            print('in pi frames (median): delta_X={0:3.2f} delta_Y={1:3.2f}'.format(med_pi_even_x-med_pi_odd_x,med_pi_even_y-med_pi_odd_y))
            print('in 0  frames (std): sig_delta_X={0:3.2f} sig_delta_Y={1:3.2f}'.format(std_0_even_minus_odd_x,std_0_even_minus_odd_y))
            print('in pi frames (std): sig_delta_X={0:3.2f} sig_delta_Y={1:3.2f}'.format(std_pi_even_minus_odd_x,std_pi_even_minus_odd_y))

            print('in 0  frames: (all values): delta_X={0} delta_Y={1}'.format(dico_center_x[self._frameTypes[0]]-dico_center_x[self._frameTypes[1]],dico_center_y[self._frameTypes[0]]-dico_center_y[self._frameTypes[1]]))
            print('in pi frames: (all values): delta_X={0} delta_Y={1}'.format(dico_center_x[self._frameTypes[2]]-dico_center_x[self._frameTypes[3]],dico_center_y[self._frameTypes[2]]-dico_center_y[self._frameTypes[3]]))
    
            self._centerx.append(dico_center_x)
            self._centery.append(dico_center_y)
        return

    def computeStokes(self,beamShift=False,save=True,rowMedianSubtraction=False,\
                      columnMedianSubtraction=False):
        """
        Populates the attributes _even_minus_odd and _even_plus_odd of the class
        by looping over the frames from the 2 cameras and the 2 phases (0 and pi), 
        and subtracting/summing the odd and even frames. 
        Optionnally it can correct the beam shift 
        Input:
            - beamshift: True to correct for the beamshiit
            - center: array of size (2,2) where center[:,0] is the X center of 
            cam1 and cam2, and center[:,1] is the X center of cam1 and cam2
            - save: True to save the fits files corresponding to even_minus_odd,
            even_plus_odd files (without and with recentering).
        Output:
            nothing
        """
        self.findCenter(method=self.recenter)
        self._even_plus_odd=[]
        self._even_minus_odd=[]
        max_beam_shift = 2
        max_shift = 20
        bias_even = np.zeros([self._rowNb,self._columnNb])
        bias_odd = np.zeros([self._rowNb,self._columnNb])
        for icam,cam in enumerate(self._cameras):
            even_minus_odd = np.zeros([self._rowNb,self._columnNb],dtype=float)
            even_plus_odd = np.zeros([self._rowNb,self._columnNb],dtype=float)
            for iphase,phase in enumerate(['0','pi']):
                
                if self._masterBias!= None and self._biasOnlyI:
#            phase0Even = np.roll(phase0Even,dither[1],axis=1) # y axis (INS3 POS4 POS/2 for cam1 and INS3 POS7 POS/2 for cam2)
#            phase0Even = np.roll(phase0Even,dither[0],axis=2) # x axis (INS3 POS3 POS for cam1 and INS3 POS6 POS for cam2)
#            phase0Odd = np.roll(phase0Odd,dither[1],axis=1) # y axis (INS3 POS4 POS/2 for cam1 and INS3 POS7 POS/2 for cam2)
#            phase0Odd = np.roll(phase0Odd,dither[0],axis=2) # x axis (INS3 POS3 POS for cam1 and INS3 POS6 POS for cam2)
#            phasePiEven = np.roll(phasePiEven,dither[1],axis=1) # y axis (INS3 POS4 POS/2 for cam1 and INS3 POS7 POS/2 for cam2)
#            phasePiEven = np.roll(phasePiEven,dither[0],axis=2) # x axis (INS3 POS3 POS for cam1 and INS3 POS6 POS for cam2)
#            phasePiOdd = np.roll(phasePiOdd,dither[1],axis=1) # y axis (INS3 POS4 POS/2 for cam1 and INS3 POS7 POS/2 for cam2)
#            phasePiOdd = np.roll(phasePiOdd,dither[0],axis=2) # x axis (INS3 POS3 POS for cam1 and INS3 POS6 POS for cam2)                    
                    if icam==0:
                        bias_even = self._masterBias._masterFrame_cam1[phase+'_even']
                        bias_odd  = self._masterBias._masterFrame_cam1[phase+'_odd']
                    else:
                        bias_even = self._masterBias._masterFrame_cam2[phase+'_even']
                        bias_odd  = self._masterBias._masterFrame_cam2[phase+'_odd']
                    bias_even = np.roll(bias_even,self.dither[cam][1],axis=0)
                    bias_even = np.roll(bias_even,self.dither[cam][0],axis=1)
                    bias_odd = np.roll(bias_odd,self.dither[cam][1],axis=0)
                    bias_odd = np.roll(bias_odd,self.dither[cam][0],axis=1)
                
                even_minus_odd_cube = self._frames[cam][phase+'_even'] - self._frames[cam][phase+'_odd']
                even_plus_odd_cube  = (self._frames[cam][phase+'_even']-bias_even) + (self._frames[cam][phase+'_odd'] - bias_odd) 
                if save:
                    HDU_even_minus_odd = fits.PrimaryHDU(even_minus_odd_cube,header=self._header)
                    HDU_even_minus_odd.writeto(os.path.join(self._pathReduc,self._name+'_cube_{0:s}_even_minus_odd_cam{1:d}.fits'.format(phase,cam)),clobber=True,output_verify='ignore')
                    HDU_even_plus_odd = fits.PrimaryHDU(even_plus_odd_cube,header=self._header)
                    HDU_even_plus_odd.writeto(os.path.join(self._pathReduc,self._name+'_cube_{0:s}_even_plus_odd_cam{1:d}.fits'.format(phase,cam)),clobber=True,output_verify='ignore')
                if beamShift:
                    # here we keep the odd frames as is and we shift the even.
                    beam_shift_dx_array = self._centerx[icam][phase+'_even']-self._centerx[icam][phase+'_odd']
                    beam_shift_dy_array = self._centery[icam][phase+'_even']-self._centery[icam][phase+'_odd']
                    validShift = self._is_below_max_shift(beam_shift_dx_array,beam_shift_dy_array,max_beam_shift)
                    if np.sum(~validShift)>0:
                        beam_shift_dx_array[~validShift]=0
                        beam_shift_dy_array[~validShift]=0
                        print('Warning {0:d} beam shift corrections were above {1:d}px. We discarded these shifts'.format(np.sum(~validShift)>0,max_beam_shift))
                    self._frames[cam][phase+'_even'] = self._shift(self._frames[cam][phase+'_even'],\
                                -beam_shift_dx_array,-beam_shift_dy_array)
                    frame_tmp_even_biasSubtracted = self._shift(self._frames[cam][phase+'_even']-bias_even,\
                                -beam_shift_dx_array,-beam_shift_dy_array)
                    dx_array = self._centerx[icam][phase+'_odd'] -  self._columnNb//2
                    dy_array = self._centery[icam][phase+'_odd'] -  self._rowNb//2
                    if np.sum(validShift)>0:
                        self._mean_beamshift[icam,iphase,0] = np.mean((self._centerx[icam][phase+'_even']-self._centerx[icam][phase+'_odd'])[validShift])
                        self._mean_beamshift[icam,iphase,1] = np.mean((self._centery[icam][phase+'_even']-self._centery[icam][phase+'_odd'])[validShift])        
                        self._std_beamshift[icam,iphase,0] = np.std((self._centerx[icam][phase+'_even']-self._centerx[icam][phase+'_odd'])[validShift])
                        self._std_beamshift[icam,iphase,1] = np.std((self._centery[icam][phase+'_even']-self._centery[icam][phase+'_odd'])[validShift])                    
                    print('Beam shift correction for cam{0:d} {1:s} ({2:d} frames): delta_X={3:.2f}px (+/-{4:.2f}) delta_Y={5:.2f}px (+/-{6:.2f})'.format(\
                          cam,phase,len(beam_shift_dx_array),\
                          self._mean_beamshift[icam,iphase,0],self._std_beamshift[icam,iphase,0],\
                          self._mean_beamshift[icam,iphase,1],self._std_beamshift[icam,iphase,1]))
                    even_minus_odd_cube = self._frames[cam][phase+'_even'] - self._frames[cam][phase+'_odd']
                    even_plus_odd_cube  = frame_tmp_even_biasSubtracted + (self._frames[cam][phase+'_odd']-bias_odd)
                else:
                    dx_array = (self._centerx[icam][phase+'_even']+self._centerx[icam][phase+'_odd'])/2. -  self._columnNb//2
                    dy_array = (self._centery[icam][phase+'_even']+self._centery[icam][phase+'_odd'])/2. -  self._rowNb//2
                dx_array_for_check = (self._centerx[icam][phase+'_even']+self._centerx[icam][phase+'_odd'])/2. -  self._guess_xy[icam,0]
                dy_array_for_check = (self._centery[icam][phase+'_even']+self._centery[icam][phase+'_odd'])/2. -  self._guess_xy[icam,1]                
                good_frames = self._is_below_max_shift(dx_array_for_check,dy_array_for_check,max_shift)
                if np.sum(~good_frames)>0:
                    dx_array[~good_frames]=0
                    dy_array[~good_frames]=0
                    bad_indices = [i for i,val in enumerate(good_frames) if ~val]
                    bad_indices_list = '/'.join(map(str,bad_indices))
                    print('Warning frames {0:s} have shifts above {1:d}px. We discarded these frames'.format(bad_indices_list,max_shift))
                print('Shifting the frame by {0:s} in X and {1:s} in Y'.format(';'.join(\
                      ['{0:4.2f}'.format(n) for n in -dx_array]),';'.join(['{0:4.2f}'.format(n) for n in -dy_array])))

                even_minus_odd_cube_centered = self._shift(even_minus_odd_cube,-dx_array,-dy_array,verbose=True)
                even_plus_odd_cube_centered = self._shift(even_plus_odd_cube,-dx_array,-dy_array,verbose=True)
                if rowMedianSubtraction:
                        even_minus_odd_cube_centered = imtools.subtract_median(even_minus_odd_cube_centered,row=True,column=False)
                        even_plus_odd_cube_centered  = imtools.subtract_median(even_plus_odd_cube_centered,row=True,column=False)
                if columnMedianSubtraction:
                    even_minus_odd_cube_centered = imtools.subtract_median(even_minus_odd_cube_centered,row=False,column=True)
                    even_plus_odd_cube_centered  = imtools.subtract_median(even_plus_odd_cube_centered,row=False,column=True)                    
                if np.any(~np.isfinite(even_minus_odd_cube_centered)):
                    print('problem with nan during recentering')
                    raise ValueError('Problem with nan during recentering')                    
                if save:
                    HDU_even_minus_odd = fits.PrimaryHDU(even_minus_odd_cube_centered,header=self._header)
                    HDU_even_minus_odd.writeto(os.path.join(self._pathReduc,self._name+'_cube_{0:s}_even_minus_odd_cam{1:d}_recentered.fits'.format(phase,cam)),clobber=True,output_verify='ignore')
                    HDU_even_plus_odd = fits.PrimaryHDU(even_plus_odd_cube_centered,header=self._header)
                    HDU_even_plus_odd.writeto(os.path.join(self._pathReduc,self._name+'_cube_{0:s}_even_plus_odd_cam{1:d}_recentered.fits'.format(phase,cam)),clobber=True,output_verify='ignore')
                even_minus_odd = even_minus_odd +  np.median(even_minus_odd_cube_centered[good_frames,:,:],axis=0)/2.
                even_plus_odd  = even_plus_odd  +  np.median(even_plus_odd_cube_centered[good_frames,:,:],axis=0)/2.
            self._even_minus_odd.append(even_minus_odd)                    
            self._even_plus_odd.append(even_plus_odd)                    
        return
       
    def _is_below_max_shift(self,shift_x,shift_y,threshold):
        """
        Check whether a value in one of the 2 input arrays is above the threshold 
        in absolute value and if yes replace it by 0.
        Returns a boolean array with the True for good frames
        """
        abs_shift_x = np.abs(shift_x)
        abs_shift_y = np.abs(shift_y)        
        goodValues = np.logical_and(abs_shift_x<threshold,abs_shift_y<threshold)
        return goodValues

    def _print_beam_shift_statistics(self):
        """
        Displays the mean beamshift with uncertainty for the 2 cameras, the 2 phases
        (0 or pi) and along the x and y directions.
        It returns 2 arrays for the mean and uncertainty (3 dimensions: camera,phase,and direction)
        """
        for icam,cam in enumerate(self._cameras):
            for iphase,phase in enumerate(['0','pi']):
                print('Beam shift correction for cam{0:d} {1:s} frames: delta_X={2:.2f}px (+/-{3:.2f}) delta_Y={4:.2f}px (+/-{5:.2f})'.format(\
                     cam,phase,\
                     self._mean_beamshift[icam,iphase,0],self._std_beamshift[icam,iphase,0],\
                     self._mean_beamshift[icam,iphase,1],self._std_beamshift[icam,iphase,1]))    
        return self._mean_beamshift,self._std_beamshift
        
    def writeCube(self,allFrames=True):
        """
        Writes the individual frames in the form of a cube of images.
        We can choose to write one cube for each frame type (0_even, 0_odd, pi_even ,pi_odd) 
        using the keyword allFrames=True
        or to sum the 0 and pi cubes together with allFrames=False
        Optional input:
        - allFrames: boolean to indicate wether all 4 types of frames must be saved
                    ('pi_odd', 'pi_even','0_even', '0_odd' frames.) or only the
                    odd and even frames
        """
        if allFrames == False:
            for cam in self.getCameras():
                cube_even = self._frames[cam]['0_even']+self._frames[cam]['pi_even']
                cube_odd = self._frames[cam]['0_odd']+self._frames[cam]['pi_odd']
                cubeHDU_even = fits.PrimaryHDU(cube_even,header=self._header)
                cubeHDU_even.writeto(os.path.join(self._pathReduc,self._name+'_cube_even_cam{0:d}.fits'.format(cam)),overwrite=True,output_verify='ignore')
                cubeHDU_odd = fits.PrimaryHDU(cube_odd,header=self._header)
                cubeHDU_odd.writeto(os.path.join(self._pathReduc,self._name+'_cube_odd_cam{0:d}.fits'.format(cam)),overwrite=True,output_verify='ignore')
        else:
            for cam in self.getCameras():
                for k in self._frameTypes:
                    cubeHDU = fits.PrimaryHDU(self._frames[cam][k],header=self._header)
                    cubeHDU.writeto(os.path.join(self._pathReduc,self._name+'_cube_{0:s}_cam{1:d}.fits'.format(k,cam)),clobber=True,output_verify='ignore')

if __name__=='__main__':
    pathRoot='/Volumes/DATA/JulienM/HD106906_ZIMPOL'
    pathRawCalib=os.path.join(pathRoot,'raw_calib')
    pathReducCalib=os.path.join(pathRoot,'calib')
    fileNameBias='SPHER.2015-07-21T14:42'
    masterBias=ZimpolMasterBias(pathRawCalib,pathReducCalib,fileNameBias,name='bias')
    masterBias.collapseFrames()

    fileNameFlat='flat'
    masterFlat=ZimpolMasterFlat(pathRawCalib,pathReducCalib,fileNameFlat,
                                name='flat',masterBias=masterBias)
    masterFlat.buildMasterFrame()
    masterFlat.write(masterOnly=True)
    
    pathRawScience='raw_science'
    pathReducScience='reduc_science'
    fileNameHWP0='SPHER.2015-07-20T23:07:11.969.fits'
    fileNameHWP45='SPHER.2015-07-20T23:12:31.265.fits'
#    scienceHWP0=ZimpolScienceCube(pathRoot,pathRawScience,pathReducScience,fileNameHWP0,\
#        name='science_HWP0_noCorrection')
#    scienceHWP0.buildMasterFrame()
#    scienceHWP0.writeCube()
    
#    scienceHWP0=ZimpolScienceCube(pathRoot,pathRawScience,pathReducScience,fileNameHWP0,\
#        name='science_HWP0_BiasCorrected',masterBias=masterBias)
#    scienceHWP0.subtractBias()
#    scienceHWP0.buildMasterFrame()
#    scienceHWP0.writeCube()

    scienceHWP0=ZimpolScienceCube(pathRawScience,pathReducScience,fileNameHWP0,\
        name='science_HWP0_BiasFlatCorrected',masterBias=masterBias,masterFlat=masterFlat)
    scienceHWP0.subtractBias()
    scienceHWP0.divideByFlat()
    scienceHWP0.correctBadPixels()
    scienceHWP0.buildMasterFrame()
    scienceHWP0.writeCube()


#    cube = scienceHWP0._cube_cam1['0_even']
#    masked_array = sigma_clip(cube,3,1,axis=0)