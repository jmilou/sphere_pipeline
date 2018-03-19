#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 09:51:17 2017

@author: jmilli
"""

import numpy as np
from astropy.io import fits
import os
#from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from zimpolDataHandler import ZimpolDataHandler
from zimpolScienceCube import ZimpolScienceCube
from polarimetricCycle import PolarimetricCycle
from rebin import rebin2d
from image_tools import distance_array,angle_array
from rotation_images import frame_rotate
from astropy.io import ascii
#import radial_data as rd

class ZimpolSetOfFiles(ZimpolDataHandler):
    """
    ZimpolSetOfFiles is collection of Zimpol files. The object will sort the files
    by type (science, calib), make sure the science files are in the same setup and
    that the calib files match the science files before grouping the files by
    polarimetric cycles to process each polar.cycle independently.
    Common attributes with ZimpolDataHandler:
        - _pathRoot: the absolute path where the reduction is performed
        - _pathRaw: the absolute path where the raw files are stored
        - _pathReduc: the absolute path where the reduced files
                      are stored
        - _fileNames: the list of filenames. 
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
    Specific attributes to ZimpolSetOfFiles
        - _scienceFilesIndices
        - recenter
        - beamShift
        - _biasOnlyI
    Common methods to ZimpolDataHandler:
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
    Specific methods to ZimpolMasterFile:
    """
    
    def __init__(self,pathRaw,pathReduc,fileNames,badPixelMap=None,\
        name='zimpol_file',masterBias=None,masterFlat=None,\
        badPixel=True,recenter='default',beamShift=False,center=None,\
        biasOnlyI=False,fft=True):
        """
        Constructor of the ZimpolSetOfFiles object. 
        Input:
            - pathRoot: the absolute path where the reduction is performed
            - pathRaw: the relative path (from pathRoot) where the raw files are stored
            - pathReduc: the relative path (from pathRoot) where the reduced files
                         are stored
            - fileNames: the list of filenames. It can be either a string 
                        with the general start of the file names, e.g. 'SPHERE_ZIMPOL_', 
                        or a list of complete filenames
            - badPixelMap: a map of bad pixel (bad=1, good=0) (optional) 
        """
        ZimpolDataHandler.__init__(self,pathRaw,pathReduc,fileNames,name)
        self._scienceFilesIndices = self._get_id_from_dpr_catg('SCIENCE')
        self._calibFilesIndices = self._get_id_from_dpr_catg('CALIB')
        self._setup_keywords = ['HIERARCH ESO DET NDIT', \
                    'HIERARCH ESO DET READ CURNAME','HIERARCH ESO DET DIT1', \
                    'HIERARCH ESO DPR TECH', 'HIERARCH ESO DPR CATG',\
                    'HIERARCH ESO INS3 OPTI2 NAME',\
                    'HIERARCH ESO INS3 OPTI5 NAME','HIERARCH ESO INS3 OPTI6 NAME',\
                    'HIERARCH ESO INS4 DROT2 POSANG','HIERARCH ESO INS4 DROT2 MODE']
        print('They are {0:d} science files'.format(len(self._scienceFilesIndices)))
        print('They are {0:d} calib files'.format(len(self._calibFilesIndices)))
        print('Setup')
        for keyword in self._setup_keywords :
            setup = set(self.getKeywords()[keyword][i] for i in self._scienceFilesIndices)
            if len(setup) != 1:
                raise TypeError('The keyword {0:s} must be identical for all science files'.format(keyword))
            print('{0:s}: {1:s}'.format(keyword,str(setup.pop())))
        self._nbPolarCycles,self._nbFramesPerHWPPos = self._get_nb_polar_cycles()
        print('Number of polarimetric cycles: {0:d}'.format(self._nbPolarCycles))
        print('Number of frames per HWP position {0:d}'.format(self._nbFramesPerHWPPos))      
        self._biasOnlyI = biasOnlyI
        self._masterBias=masterBias
        self._masterFlat=masterFlat    
        self._badPixelCorrection=badPixel
        self.recenter=recenter
        self.beamShift=beamShift
        self._I = [] # a list of cubes for the 2 cameras
        self._Q = []
        self._U = []
        self._fft = fft
        if center==None:
            self._guess_center_xy=np.asarray([[self._columnNb//2,self._rowNb//2],[self._columnNb//2,self._rowNb//2]])
        else:
            if np.asarray(center).ndim == 1:
                self._guess_center_xy = np.asarray([center,center])
            elif np.asarray(center).ndim == 2:
                self._guess_center_xy =np.asarray(center)
            else:
                raise TypeError('The center guess has the wrong shape. It should be an [x,y] array or a [[x_cam1,y_cam1],[x_cam2,y_cam2]] array')

    def _get_nb_polar_cycles(self):
        """
        Returns the number of polarimetric cycles and the number of frames 
        for each HWP position
        """
        hwp_posang = [self.getKeywords()['HIERARCH ESO INS4 DROT3 POSANG'][i] for i in self._scienceFilesIndices]
        if np.mod(len(hwp_posang),4) != 0:
            raise TypeError('The number of science files should be a multiple of 4 but is {0:d} '.format(len(hwp_posang)))
        nb_frames_per_hwp_pos = 1
        for i,posang in enumerate(hwp_posang[1:]):
            if posang == hwp_posang[0]:
                nb_frames_per_hwp_pos += 1
            else:
                break
        if np.mod(len(hwp_posang),4*nb_frames_per_hwp_pos) != 0:
            raise TypeError('The number of science files should be a multiple of {0:d} but is {1:d} '.format(\
                            4*nb_frames_per_hwp_pos,len(hwp_posang)))
        nb_cycles = len(hwp_posang)/(4*nb_frames_per_hwp_pos)
        return nb_cycles,nb_frames_per_hwp_pos

    def _get_id_polar_cycle(self,polarCycleId):
        """
        Returns a list of indices of the science frames corresponding to the given 
        polar. cycle. 
        Input:
            - index of the polarimetric cycle (it starts at 0)
        Output:
            - indices of the frames corresponding to that polar. cycle
        """
        idScienceFrames = np.arange(4*self._nbFramesPerHWPPos*(polarCycleId),4*self._nbFramesPerHWPPos*(polarCycleId+1))
        id_frames = [self._scienceFilesIndices[i] for i in idScienceFrames]
        return id_frames

    def reduce_one_polar_cycle(self,polarCycleId):
        """
        Builds the +Q,-Q,+U and -U ZimpolScienceCube objects for a given polar cycle.
        It combines the frames into intensity and Stokes parameters, saves them to disk 
        and returns the 4 objects.
        Input:
            - index of the polarimetric cycle (it starts at 0)
        Output:
            - PolarimetriCycle object
        """
        id_frames = self._get_id_polar_cycle(polarCycleId)
#        print(id_frames)
        id_plusQ_frames = id_frames[0:self._nbFramesPerHWPPos]
        id_minusQ_frames = id_frames[self._nbFramesPerHWPPos:self._nbFramesPerHWPPos*2]
        id_plusU_frames = id_frames[self._nbFramesPerHWPPos*2:self._nbFramesPerHWPPos*3]
        id_minusU_frames = id_frames[self._nbFramesPerHWPPos*3:]

        fileNamesPlusQFrames = [self._fileNames[i] for i in id_plusQ_frames]
        namePlusQ = 'cycle{0:02d}_HWP{1:04d}'.format(polarCycleId,np.int(self.getKeywords()['HIERARCH ESO INS4 DROT3 POSANG'][id_plusQ_frames[0]])) 
        plusQCube = ZimpolScienceCube(self._pathRaw,self._pathReduc,fileNamesPlusQFrames,\
                 name=namePlusQ,masterBias=self._masterBias,masterFlat=self._masterFlat,\
                 badPixel=self._badPixelCorrection,guess_xycenter=self._guess_center_xy,\
                 biasOnlyI= self._biasOnlyI,recenter=self.recenter,fft=self._fft)

        fileNamesMinusQFrames = [self._fileNames[i] for i in id_minusQ_frames]
        nameMinusQ = 'cycle{0:02d}_HWP{1:04d}'.format(polarCycleId,np.int(self.getKeywords()['HIERARCH ESO INS4 DROT3 POSANG'][id_minusQ_frames[0]])) 
        minusQCube = ZimpolScienceCube(self._pathRaw,self._pathReduc,fileNamesMinusQFrames,\
                 name=nameMinusQ,masterBias=self._masterBias,masterFlat=self._masterFlat,\
                 badPixel=self._badPixelCorrection,guess_xycenter=self._guess_center_xy,\
                 biasOnlyI= self._biasOnlyI,recenter=self.recenter,fft=self._fft)

        fileNamesPlusUFrames = [self._fileNames[i] for i in id_plusU_frames]
        namePlusU = 'cycle{0:02d}_HWP{1:04d}'.format(polarCycleId,np.int(self.getKeywords()['HIERARCH ESO INS4 DROT3 POSANG'][id_plusU_frames[0]])) 
        plusUCube = ZimpolScienceCube(self._pathRaw,self._pathReduc,fileNamesPlusUFrames,\
                 name=namePlusU,masterBias=self._masterBias,masterFlat=self._masterFlat,\
                 badPixel=self._badPixelCorrection,guess_xycenter=self._guess_center_xy,\
                 biasOnlyI= self._biasOnlyI,recenter=self.recenter,fft=self._fft)

        fileNamesMinusUFrames = [self._fileNames[i] for i in id_minusU_frames]
        nameMinusU = 'cycle{0:02d}_HWP{1:04d}'.format(polarCycleId,np.int(self.getKeywords()['HIERARCH ESO INS4 DROT3 POSANG'][id_minusU_frames[0]])) 
        minusUCube = ZimpolScienceCube(self._pathRaw,self._pathReduc,fileNamesMinusUFrames,\
                 name=nameMinusU,masterBias=self._masterBias,masterFlat=self._masterFlat,\
                 badPixel=self._badPixelCorrection,guess_xycenter=self._guess_center_xy,\
                 biasOnlyI= self._biasOnlyI,recenter=self.recenter,fft=self._fft)

        polarimetricCycle = PolarimetricCycle(plusQCube,minusQCube,plusUCube,minusUCube,\
                                              beamShift=self.beamShift)
        
        for icam,cam in enumerate(plusQCube.getCameras()):
            fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_I_cam{1:d}.fits'.format(polarCycleId,cam)),polarimetricCycle._I[icam],plusQCube._header,clobber=True,output_verify='ignore')
            fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_Q_cam{1:d}.fits'.format(polarCycleId,cam)),polarimetricCycle._Q[icam],plusQCube._header,clobber=True,output_verify='ignore')
            fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_U_cam{1:d}.fits'.format(polarCycleId,cam)),polarimetricCycle._U[icam],plusUCube._header,clobber=True,output_verify='ignore')

#            
#        fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_I_cam1.fits'.format(polarCycleId)),polarimetricCycle._I_cam1,plusQCube._header,clobber=True,output_verify='ignore')
#        fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_I_cam2.fits'.format(polarCycleId)),polarimetricCycle._I_cam2,plusQCube._header,clobber=True,output_verify='ignore')
#
#        fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_Q_cam1.fits'.format(polarCycleId)),polarimetricCycle._Q_cam1,plusQCube._header,clobber=True,output_verify='ignore')
#        fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_Q_cam2.fits'.format(polarCycleId)),polarimetricCycle._Q_cam2,plusQCube._header,clobber=True,output_verify='ignore')
#
#        fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_U_cam1.fits'.format(polarCycleId)),polarimetricCycle._U_cam1,plusUCube._header,clobber=True,output_verify='ignore')
#        fits.writeto(os.path.join(self._pathReduc,'cycle{0:02d}_U_cam2.fits'.format(polarCycleId)),polarimetricCycle._U_cam2,plusUCube._header,clobber=True,output_verify='ignore')
#
        return polarimetricCycle

    def combinePolarimetricCycles(self,method='mean',readonly=False):
        """
        Combines the different polarimetric cycles and creates the Stokes I, Q and U.
        If the combinationwas already made and the output folder already contains 
        the fits files Q_cam1.fits... then it can also be read only with the keyword 
        readonly=True
        Input:
            - method: the method to build the master frame. By default, uses a 
             median. Accepts 'sigma_clip' (sigma clipping of sigma=3 with 1 iteration)
             or 'mean'
        Output:
            - list of PolarimetricCycles objects
        """
        if readonly==False:
            listPolarCycles = []
            for i in range(self._nbPolarCycles):
                listPolarCycles.append(self.reduce_one_polar_cycle(i))              
            if self.beamShift:
                # these 2 arrays will store the mean beam shift over the different cycles.
                mean_beamshift = np.ndarray((4,2,2,2))*0.#(4 dimensions: HWP,camera,phase,and direction)
                std_beamshift = np.ndarray((4,2,2,2))*0.
                for i in range(self._nbPolarCycles):
                    print('Beam shift summary')
                    print('Polarization cycle {0:d}'.format(i))
                    mean_beamshift_cycle,std_beamshift_cycle = listPolarCycles[i].print_beam_shift_statistics()
                    mean_beamshift +=mean_beamshift_cycle/self._nbPolarCycles
                    std_beamshift  += std_beamshift_cycle/self._nbPolarCycles
                for icam,cam in enumerate(self.getCameras()):
                    description = ['+Q_even','+Q_odd','-Q_even','-Q_odd','+U_even','+U_odd','-U_even','-U_odd']
                    beamshift_to_write = np.ndarray((8,4))
                    beamshift_to_write[0:2,0:2] = mean_beamshift[0,icam,:,:]
                    beamshift_to_write[2:4,0:2] = mean_beamshift[1,icam,:,:]
                    beamshift_to_write[4:6,0:2] = mean_beamshift[2,icam,:,:]
                    beamshift_to_write[6:,0:2]  = mean_beamshift[3,icam,:,:]
                    beamshift_to_write[0:2,2:] = std_beamshift[0,icam,:,:]
                    beamshift_to_write[2:4,2:] = std_beamshift[1,icam,:,:]
                    beamshift_to_write[4:6,2:] = std_beamshift[2,icam,:,:]
                    beamshift_to_write[6:,2:]  = std_beamshift[3,icam,:,:]
                    ascii.write([description,beamshift_to_write[:,0],beamshift_to_write[:,1],\
                                 beamshift_to_write[:,2],beamshift_to_write[:,3]],\
                                os.path.join(self._pathReduc,'beamshift_correction_cam{0:d}.csv'.format(icam+1)),\
                                names=['camera','x','y','err_x','err_y'],format='csv')        
    
            cube_I = []
            cube_Q = []
            cube_U = []
            for icam,cam in enumerate(self.getCameras()):
                cube_I_cam = np.zeros((self._nbPolarCycles,self._rowNb,self._columnNb))
                cube_Q_cam = np.zeros((self._nbPolarCycles,self._rowNb,self._columnNb))
                cube_U_cam = np.zeros((self._nbPolarCycles,self._rowNb,self._columnNb))
                for i in range(self._nbPolarCycles):
                    cube_I_cam[i,:,:] = listPolarCycles[i]._I[icam]
                    cube_Q_cam[i,:,:] = listPolarCycles[i]._Q[icam]
                    cube_U_cam[i,:,:] = listPolarCycles[i]._U[icam]
                cube_I.append(cube_I_cam)
                cube_Q.append(cube_Q_cam)
                cube_U.append(cube_U_cam)            
            if method == 'mean':
                f=np.nanmean
            elif method == 'median':
                f=np.nanmedian
            else:
                raise Exception('Combination method not available: {0:s}'.format(method))            
            self._I.append(np.flipud(f(cube_I[0],axis=0))) # cam 1
            self._I.append(np.flipud(np.fliplr(f(cube_I[1],axis=0)))) # for cam 2
            self._Q.append(-np.flipud(f(cube_Q[0],axis=0)))# cam 1
            self._Q.append(np.flipud(np.fliplr(f(cube_Q[1],axis=0))))# for cam 2
            self._U.append(-np.flipud(f(cube_U[0],axis=0))) # cam 1
            self._U.append(np.flipud(np.fliplr(f(cube_U[1],axis=0))))# for cam 2
            for icam,cam in enumerate(self.getCameras()):
                fits.writeto(os.path.join(self._pathReduc,'I_cam{0:d}.fits'.format(cam)),self._I[icam],listPolarCycles[0]._plusQ._header,clobber=True,output_verify='ignore')
                fits.writeto(os.path.join(self._pathReduc,'Q_cam{0:d}.fits'.format(cam)),self._Q[icam],listPolarCycles[0]._plusQ._header,clobber=True,output_verify='ignore')
                fits.writeto(os.path.join(self._pathReduc,'U_cam{0:d}.fits'.format(cam)),self._U[icam],listPolarCycles[0]._plusU._header,clobber=True,output_verify='ignore')
            return listPolarCycles
        else:
            for icam,cam in enumerate(self.getCameras()):
                self._I.append(fits.getdata(os.path.join(self._pathReduc,'I_cam{0:d}.fits'.format(cam))))
                self._Q.append(fits.getdata(os.path.join(self._pathReduc,'Q_cam{0:d}.fits'.format(cam))))
                self._U.append(fits.getdata(os.path.join(self._pathReduc,'U_cam{0:d}.fits'.format(cam))))
            return 

    def rebinStokes(self):
        """
        rebin the Stokes parameter and align the image with the north up using 
        the keyword INS4.DROT2.POSANG
        """
        self._I_rebinned = []
        self._Q_rebinned = []
        self._U_rebinned = []
        position_angle = self.getKeywords()['HIERARCH ESO INS4 DROT2 POSANG'][0]
        print('Rebinning the pixels by a factor 2 horizontally and rotating by {0:.0f}deg to alight the north up'.format(position_angle))
        for icam,cam in enumerate(self.getCameras()):
            I_rebinned = rebin2d(self._I[icam],(self._columnNb/2,self._rowNb))
            Q_rebinned = rebin2d(self._Q[icam],(self._columnNb/2,self._rowNb))
            U_rebinned = rebin2d(self._U[icam],(self._columnNb/2,self._rowNb))
            I_rebinned = frame_rotate(I_rebinned, -position_angle)
            Q_rebinned = frame_rotate(Q_rebinned, -position_angle)
            U_rebinned = frame_rotate(U_rebinned, -position_angle)
            self._I_rebinned.append(I_rebinned)
            self._Q_rebinned.append(Q_rebinned)
            self._U_rebinned.append(U_rebinned)
            fits.writeto(os.path.join(self._pathReduc,'I_cam{0:d}_rebinned.fits'.format(cam)),self._I_rebinned[icam],clobber=True,output_verify='ignore')
            fits.writeto(os.path.join(self._pathReduc,'Q_cam{0:d}_rebinned.fits'.format(cam)),self._Q_rebinned[icam],clobber=True,output_verify='ignore')
            fits.writeto(os.path.join(self._pathReduc,'U_cam{0:d}_rebinned.fits'.format(cam)),self._U_rebinned[icam],clobber=True,output_verify='ignore')


    def correctInstrumentalPolarisation(self,mask=None,path_nosat=None,method='RMS'):
        """
        Corrects the instrumental polarisation (IP) by subtracting to Q or U a scaled version of I. 
        It generates a csv file with the correction factor applied to each Stokes and
        the level of background before and after IP correction
        Input:
            -path_nosat: if not specified, the function uses the Q,U images themselves
                        to measure the IP. If one wants to apply a given factor, then
                        one can specify a folder name containing a file 
                        'scaling_IP_correction.csv' that has among other 2 columns
                        'Q_factor' and 'U_factor' and 2 lines for cam1 and cam2
                        listing the scaling factor to be used (for instance 0.001 
                        if you want to subtract 0.001*I from Q or U)
            - mask: can be a boolean array of the size of _I_rebinned set to True
                    for the pixels to be used for the IP calculation, or a dictionary
                    with 'rin' and 'rout' as keys. 
            - method: string to indicate what should be minimized (or nulled) in 
                    the masked regions. By default 'RMS' is assumed: it minimizes the norm 2 (root mean square).
                    Can also be 'median' or 'mean'
        """
        try: 
            distarr = distance_array(self._I_rebinned[0].shape)
        except AttributeError:
            print('The Stokes were not rebinned. Doing it now...')
            self.rebinStokes()
            distarr = distance_array(self._I_rebinned[0].shape)
        if mask==None:
            if path_nosat!=None:
                print('For statistics, I am using the inner 15 pixels as mask')            
            else:
                print('No mask provided for the IP correction. I am using the inner 15 pixels to minimize the RMS')  
            mask = distarr<15
        elif isinstance(mask,dict):
            if 'rin' in mask.keys() and 'rout' in mask.keys():
                mask = np.logical_and(distarr>mask['rin'],distarr<mask['rout'])
            else:
                print('The argument of correctInstrumentalPolarisation is not a valid dictionnary for a mask. It must contain the keys "rin" and "rout"')
        self._Q_corrected = []
        self._U_corrected = []
        scaling_Q = np.ndarray(2)
        scaling_U = np.ndarray(2)
        bkg_med_Q_wo_IP = np.ndarray(2)
        bkg_med_U_wo_IP = np.ndarray(2)
        bkg_med_Q_w_IP = np.ndarray(2)
        bkg_med_U_w_IP = np.ndarray(2)
        bkg_std_Q_wo_IP = np.ndarray(2)
        bkg_std_U_wo_IP = np.ndarray(2)
        bkg_std_Q_w_IP = np.ndarray(2)
        bkg_std_U_w_IP = np.ndarray(2)
        mask_check_background_level = np.logical_and(distarr>50,distarr<240)
        for icam,cam in enumerate(self.getCameras()):
            if path_nosat!=None:
                IP_correction_filename = os.path.join(path_nosat,'scaling_IP_correction.csv')
                print('Reading the IP correction file {0:s}'.format(IP_correction_filename))
                ascii_file = ascii.read(IP_correction_filename)
                scaling_Q[icam] = ascii_file['Q_factor'][icam]
                scaling_U[icam] = ascii_file['U_factor'][icam]
            else:
                if method=='RMS':
                    scaling_Q[icam] = np.sum(self._I_rebinned[icam]*self._Q_rebinned[icam]*mask)/np.sum((self._I_rebinned[icam]*mask)**2)
                    scaling_U[icam] = np.sum(self._I_rebinned[icam]*self._U_rebinned[icam]*mask)/np.sum((self._I_rebinned[icam]*mask)**2)
                elif method=='median':
                    scaling_Q[icam] = np.median(self._Q_rebinned[icam][mask])/np.median(self._I_rebinned[icam][mask])
                    scaling_U[icam] = np.median(self._U_rebinned[icam][mask])/np.median(self._I_rebinned[icam][mask])
                elif method=='mean':
                    scaling_Q[icam] = np.mean(self._Q_rebinned[icam][mask])/np.mean(self._I_rebinned[icam][mask])
                    scaling_U[icam] = np.mean(self._U_rebinned[icam][mask])/np.mean(self._I_rebinned[icam][mask])
                else:
                    print('The Argument method of correctInstrumentalPolarisation was not understood: {0:s}'.format(method))
                    return                    
            print('Scaling factor Q cam{0:d} = {1:4.2e}'.format(cam,scaling_Q[icam]))
            Q_corrected = self._Q_rebinned[icam]-scaling_Q[icam]*self._I_rebinned[icam]
            print('Scaling factor U cam{0:d} = {1:4.2e}'.format(cam,scaling_U[icam]))
            U_corrected = self._U_rebinned[icam]-scaling_U[icam]*self._I_rebinned[icam]            
            self._Q_corrected.append(Q_corrected)
            self._U_corrected.append(U_corrected)
            bkg_med_Q_wo_IP[icam] = np.median(self._Q_rebinned[icam][np.where(mask_check_background_level)])
            bkg_med_U_wo_IP[icam] = np.median(self._U_rebinned[icam][np.where(mask_check_background_level)])
            bkg_med_Q_w_IP[icam] = np.median(Q_corrected[np.where(mask_check_background_level)])
            bkg_med_U_w_IP[icam] = np.median(U_corrected[np.where(mask_check_background_level)])
            bkg_std_Q_wo_IP[icam] = np.std(self._Q_rebinned[icam][np.where(mask_check_background_level)])
            bkg_std_U_wo_IP[icam] = np.std(self._U_rebinned[icam][np.where(mask_check_background_level)])
            bkg_std_Q_w_IP[icam] = np.std(Q_corrected[np.where(mask_check_background_level)])
            bkg_std_U_w_IP[icam] = np.std(U_corrected[np.where(mask_check_background_level)])
            med_Q_wo_IP = np.median(self._Q_rebinned[icam][mask])
            med_U_wo_IP = np.median(self._U_rebinned[icam][mask])
            med_Q_w_IP = np.median(Q_corrected[mask])
            med_U_w_IP = np.median(U_corrected[mask])
            mean_Q_wo_IP = np.mean(self._Q_rebinned[icam][mask])
            mean_U_wo_IP = np.mean(self._U_rebinned[icam][mask])
            mean_Q_w_IP = np.mean(Q_corrected[mask])
            mean_U_w_IP = np.mean(U_corrected[mask])
            std_Q_wo_IP = np.std(self._Q_rebinned[icam][mask])
            std_U_wo_IP = np.std(self._U_rebinned[icam][mask])
            std_Q_w_IP = np.std(Q_corrected[mask])
            std_U_w_IP = np.std(U_corrected[mask])
            print('The median flux in the mask for Q cam{0:d} went from {1:4.2e} to {2:4.2e} after IP correction ({3:.1f}% change)'.format(cam,\
              med_Q_wo_IP,med_Q_w_IP,(med_Q_w_IP-med_Q_wo_IP)/med_Q_wo_IP*100.))
            print('The median flux in the mask for U cam{0:d} went from {1:4.2e} to {2:4.2e} after IP correction ({3:.1f}% change)'.format(cam,\
              med_U_wo_IP,med_U_w_IP,(med_U_w_IP-med_U_wo_IP)/med_U_wo_IP*100.))
            print('The mean flux in the mask for Q cam{0:d} went from {1:4.2e} to {2:4.2e} after IP correction ({3:.1f}% change)'.format(cam,\
              mean_Q_wo_IP,mean_Q_w_IP,(mean_Q_w_IP-mean_Q_wo_IP)/mean_Q_wo_IP*100.))
            print('The mean flux in the mask for U cam{0:d} went from {1:4.2e} to {2:4.2e} after IP correction ({3:.1f}% change)'.format(cam,\
              mean_U_wo_IP,mean_U_w_IP,(mean_U_w_IP-mean_U_wo_IP)/mean_U_wo_IP*100.))
            print('The RMS in the mask for Q cam{0:d} went from {1:4.2e} to {2:4.2e} after IP correction ({3:.1f}% change)'.format(cam,\
              std_Q_wo_IP,std_Q_w_IP,(std_Q_w_IP-std_Q_wo_IP)/std_Q_wo_IP*100.))
            print('The RMS in the mask for U cam{0:d} went from {1:4.2e} to {2:4.2e} after IP correction ({3:.1f}% change)'.format(cam,\
              std_U_wo_IP,std_U_w_IP,(std_U_w_IP-std_U_wo_IP)/std_U_wo_IP*100.))
            
            fits.writeto(os.path.join(self._pathReduc,'Q_cam{0:d}_IPcorrected.fits'.format(cam)),self._Q_corrected[icam],clobber=True,output_verify='ignore')
            fits.writeto(os.path.join(self._pathReduc,'U_cam{0:d}_IPcorrected.fits'.format(cam)),self._U_corrected[icam],clobber=True,output_verify='ignore')
            print('The median background in Q cam{0:d} went from {1:4.2e} to {2:4.2e} after IP correction'.format(cam,\
              bkg_med_Q_wo_IP[icam],bkg_med_Q_w_IP[icam]))
            print('The median background in U cam{0:d} went from {1:4.2e} to {2:4.2e} after IP correction'.format(cam,\
              bkg_med_U_wo_IP[icam],bkg_med_U_w_IP[icam]))
        ascii.write([np.arange(1,3),scaling_Q,scaling_U,bkg_med_Q_wo_IP,bkg_med_U_wo_IP,\
                     bkg_med_Q_w_IP,bkg_med_U_w_IP,bkg_std_Q_wo_IP,bkg_std_U_wo_IP,\
                     bkg_std_Q_w_IP,bkg_std_U_w_IP],os.path.join(self._pathReduc,'scaling_IP_correction.csv'),\
                    names=['camera','Q_factor','U_factor','bkg_med_Q_wo_IP','bkg_med_U_wo_IP',\
                           'bkg_med_Q_w_IP','bkg_med_U_w_IP','bkg_std_Q_wo_IP','bkg_std_U_wo_IP',\
                           'bkg_std_Q_w_IP','bkg_std_U_w_IP'],format='csv')        
        return            
    
    def compute_pI_polarStokes(self):
        """
        computes the polarimetric intensity wth Q^2+U^2
        """
        self._pI = []
        self._Q_phi = []
        self._U_phi = []
        theta = angle_array((self._columnNb/2,self._rowNb))
        cos2theta = np.cos(2*theta)
        sin2theta = np.sin(2*theta)
        for icam,cam in enumerate(self.getCameras()):
            try:
                Q = self._Q_corrected[icam]
                U = self._U_corrected[icam]
            except AttributeError:
                self.rebinStokes()
                Q = self._Q_rebinned[icam]
                U = self._U_rebinned[icam]
            pI = np.sqrt(Q**2+U**2)            
            self._pI.append(pI)
            fits.writeto(os.path.join(self._pathReduc,'pI_cam{0:d}.fits'.format(cam)),self._pI[icam],clobber=True,output_verify='ignore')
            Q_phi = Q*cos2theta+U*sin2theta
            U_phi = -Q*sin2theta+U*cos2theta
            self._Q_phi.append(Q_phi)
            fits.writeto(os.path.join(self._pathReduc,'Qphi_cam{0:d}.fits'.format(cam)),self._Q_phi[icam],clobber=True,output_verify='ignore')
            self._U_phi.append(U_phi)
            fits.writeto(os.path.join(self._pathReduc,'Uphi_cam{0:d}.fits'.format(cam)),self._U_phi[icam],clobber=True,output_verify='ignore')            

    def test_HWP_correction(self):
        """
        Tries different offsets of the HWP and computes Qphi and Uphi in each case.
        Plots the sum of Qphi, Uphi and norm 2 of Qphi, Uphi (central pixels only)
        to find the best value that minimizes the flux in Uphi.
        Returns the offsets array, with the cubes of Qphi and Uphi  for each offset
        """
        theta_min= -90
        theta_max=90
        theta0 = np.arange(theta_min,theta_max,1.)
        nb_theta = len(theta0)
        ny,nx = (self._columnNb/2,self._rowNb)
        theta = angle_array((ny,nx))
        distarr = distance_array((ny,nx))
        mask_polar = distarr<12
        Q_phi = np.ndarray((nb_theta,ny,nx))
        U_phi = np.ndarray((nb_theta,ny,nx))
        theta_array= np.ndarray((nb_theta,ny,nx))
        total_U_phi = np.ndarray((nb_theta))
        total_Q_phi = np.ndarray((nb_theta))
        norm2_U_phi = np.ndarray((nb_theta))
        norm2_Q_phi = np.ndarray((nb_theta))
        for i,thetai in enumerate(theta0):
            theta_array[i,:,:] = np.mod(theta-np.deg2rad(thetai),2*np.pi)
            Q_phi[i,:,:] = zimpolSetOfFiles_posang000._Q_rebinned[0]+zimpolSetOfFiles_posang000._U_rebinned[0]*np.sin(2*theta_array[i,:,:])
            U_phi[i,:,:] = -zimpolSetOfFiles_posang000._Q_rebinned[0]*np.sin(2*theta_array[i,:,:])+zimpolSetOfFiles_posang000._U_rebinned[0]*np.cos(2*theta_array[i,:,:])
            total_Q_phi[i] = np.nansum(Q_phi[i,:,:]*mask_polar)
            total_U_phi[i] = np.nansum(U_phi[i,:,:]*mask_polar)
            norm2_Q_phi[i] = np.nansum((mask_polar*Q_phi[i,:,:])**2)
            norm2_U_phi[i] = np.nansum((mask_polar*U_phi[i,:,:])**2)
        
        plt.close()
        plt.figure(1)
        plt.plot(theta0,total_Q_phi,label='total Q_phi')
        plt.plot(theta0,total_U_phi,label='total U_phi')
        plt.grid()
        plt.legend()
        
        plt.figure(2)
        plt.plot(theta0,norm2_Q_phi,label='norm2 Q_phi')
        plt.plot(theta0,norm2_U_phi,label='norm2 U_phi')
        plt.grid()
        plt.legend()
        
        index_max_total_Q = np.argmax(total_Q_phi)
        index_zero_total_U = np.argmin(np.abs(total_U_phi))
        print('Max of total Q for theta offset of {0:d}'.format(theta0[index_max_total_Q]))
        print('Zero of total U for theta offset of {0:d}'.format(theta0[index_zero_total_U]))
        
        index_max_norm2_Q = np.argmax(norm2_Q_phi)
        index_min_norm2_Q = np.argmin(norm2_U_phi)
        print('Max of norm2 Q for theta offset of {0:d}'.format(theta0[index_max_norm2_Q]))
        print('Min of norm2 U for theta offset of {0:d}'.format(theta0[index_min_norm2_Q]))
        return theta0,Q_phi,U_phi
              
if __name__=='__main__':
    pathRoot='/Volumes/MILOU_1TB_2/HR4796_zimpol_GTO'
    pathRawScience=os.path.join(pathRoot,'raw_science')
    pathRawCalib=os.path.join(pathRoot,'raw_calib')
    pathReducScience=os.path.join(pathRoot,'reduc_science')
    pathReducCalib=os.path.join(pathRoot,'reduc_calib')

#    pathRawScience_posang000=os.path.join(pathRawScience,'SlowPolarimetry_000')
#    pathReducScience_posang000=os.path.join(pathReducScience,'SlowPolarimetry_000_reduced')
#    fileNames='SPHER'    

    import pdb
    pathRawScience_posang000=os.path.join(pathRawScience,'FastPolarimetry_000')
    pathReducScience_posang000=os.path.join(pathReducScience,'FastPolarimetry_000_reduced_beamshifted')
    fileNames='SPHER.2016-05-25T01:3[4-6]*'    
    zimpolSetOfFiles_posang000 = ZimpolSetOfFiles(pathRawScience_posang000,\
        pathReducScience_posang000,fileNames,recenter='gaussianFit',beamShift=True)
#    zimpolSetOfFiles_posang000.reduce_one_polar_cycle(0)
#    zimpolSetOfFiles_posang000.reduce_one_polar_cycle(1)
    zimpolSetOfFiles_posang000.combinePolarimetricCycles()
    zimpolSetOfFiles_posang000.rebinStokes()
    zimpolSetOfFiles_posang000.correctInstrumentalPolarisation()
    zimpolSetOfFiles_posang000.compute_pI_polarStokes()

#    fits.writeto(os.path.join(zimpolSetOfFiles_posang000._pathReduc,'Q.fits'),zimpolSetOfFiles_posang000._Q_cam1+zimpolSetOfFiles_posang000._Q_cam2,clobber=True)
#    fits.writeto(os.path.join(zimpolSetOfFiles_posang000._pathReduc,'U.fits'),zimpolSetOfFiles_posang000._U_cam1+zimpolSetOfFiles_posang000._U_cam2,clobber=True)

#    zimpolSetOfFiles_posang000.reduce_one_polar_cycle(1)
