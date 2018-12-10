# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:43:25 2015

@author: jmilli
"""

import numpy as np
from astropy.io import fits
import os
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from zimpolDataHandler import ZimpolDataHandler
#from astropy.io import ascii

class ZimpolMasterFile(ZimpolDataHandler):
    """
        Common attributes with ZimpolDataHandler:
        - _pathRaw: the absolute path where the raw files are stored
        - _pathReduc: the absolute path where the reduced files
                      are stored
        - _fileNames: the list of filenames. It can be either a string 
                      with the general start of the 
                      file names, e.g. 'SPHERE_ZIMPOL_', or a list of complete filenames
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
    Specific attributes to ZimpolMasterFile
        - _frames : a list of 2 dictionnaries over the 4 frame types containing all frames        
        - _guess_center: an array of shape [2,2] with a guess center for the center of the 
            frame for cam1 and cam2.
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
    Each of those attributes is a dictionnary containing the following entries:
    'pi_odd', 'pi_even', '0_even', and '0_odd'.
        - _nbPolarCycles
        - pathReducCycle
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
        - buildMasterFrame
        - findCenter
        - write
        - collapseFrames
        - getStatistics
    """
    
    def __init__(self,pathRaw,pathReduc,fileNames,badPixelMap=None,\
        name='zimpol_file',subtractDark=False,badframes=None):
        """
        Constructor of the ZimpolMasterFile object. It instantiates the master frame
        with nan values. It takes the same input 
        as ZimpolDataHandler, plus an optional bad pixel map.
        Input:
            - pathRaw: the absolute path where the raw files are stored
            - pathReduc: the absolute path where the reduced files
                         are stored
            - fileNames: the list of filenames. It can be either a string 
                        with the general start of the file names, e.g. 'SPHERE_ZIMPOL_', 
                        or a list of complete filenames
            - badPixelMap: a map of bad pixel (bad=1, good=0) (optional) 
        """
        ZimpolDataHandler.__init__(self,pathRaw,pathReduc,fileNames,name)
        self._frames = self.extractFramesFromFiles(subtractDark=subtractDark)
#        self.findCenter()
        self._masterFrame_cam1 = {}
        self._badPixelMap_cam1 = {}
        self._rmsMap_cam1 = {}
        self._weightMap_cam1 = {}
        self._masterFrame_cam2 = {}
        self._badPixelMap_cam2 = {}
        self._rmsMap_cam2 = {}
        self._weightMap_cam2 = {}
        for k in self._frameTypes:
            self._masterFrame_cam1[k]=np.empty([self._rowNb,self._columnNb],dtype=float)
            self._badPixelMap_cam1[k]=np.empty([self._rowNb,self._columnNb],dtype=int)
            self._weightMap_cam1[k]=np.empty([self._rowNb,self._columnNb],dtype=int)
            self._rmsMap_cam1[k]=np.empty([self._rowNb,self._columnNb],dtype=float)
            self._masterFrame_cam1[k].fill(np.nan)
            self._badPixelMap_cam1[k].fill(0)
            self._weightMap_cam1[k].fill(0)
            self._rmsMap_cam1[k].fill(np.nan)
            self._masterFrame_cam2[k]=np.empty([self._rowNb,self._columnNb],dtype=float)
            self._badPixelMap_cam2[k]=np.empty([self._rowNb,self._columnNb],dtype=int)
            self._weightMap_cam2[k]=np.empty([self._rowNb,self._columnNb],dtype=int)
            self._rmsMap_cam2[k]=np.empty([self._rowNb,self._columnNb],dtype=float)
            self._masterFrame_cam2[k].fill(np.nan)
            self._badPixelMap_cam2[k].fill(0)
            self._weightMap_cam2[k].fill(0)
            self._rmsMap_cam2[k].fill(np.nan)
            

    def collapseFrames(self,method='mean',debug=False,frames=None,subtractDark=False):
        """
        Collapse all the frames and populates the attributes _masterFrame, 
        _rmsMap, weightMap.
        Input:
            - method: the method to build the master frame. By default, uses a 
             median. Accepts 'sigma_clip' (sigma clipping of sigma=3 with 1 iteration)
             or 'mean'
        Optional input:
             - debug: booleean (False by default). In order to check for deviant 
                      frames. Can be usefulf for instance for the flat frames because
                      the first frame is sometimes with a lower flux.
             - frames:  list of frames to collapse (by defalut, it uses all frames, e.g.
                     range(0,NDIT/2). If the first frame is bad, use range(1,NDIT/2))
        Output: nothing
        """
        if 'header' in self._frames.keys():
            self._header = self._frames['header']
        dico_cam1 = self._frames[1]
        dico_cam2 = self._frames[2]
        if frames == None:
            frames=range(dico_cam1[self._frameTypes[0]].shape[0])
        else:
            print('Collapse of a selection of frames only')            
            print(frames)
        if debug:
            deviation_cam1={}
            deviation_cam2={}
        for k in self._frameTypes:
            if method == 'median':
                self._masterFrame_cam1[k] = np.nanmedian(dico_cam1[k][frames,:,:],axis=0)
                self._rmsMap_cam1[k] = np.nanstd(dico_cam1[k][frames,:,:],axis=0)
                self._weightMap_cam1[k] = np.nansum(np.isfinite(dico_cam1[k]),axis=0)
                self._masterFrame_cam2[k] = np.nanmedian(dico_cam2[k][frames,:,:],axis=0)
                self._rmsMap_cam2[k] = np.nanstd(dico_cam2[k][frames,:,:],axis=0)
                self._weightMap_cam2[k] = np.nansum(np.isfinite(dico_cam2[k][frames,:,:]),axis=0)
            elif method == 'mean':
                self._masterFrame_cam1[k] = np.nanmean(dico_cam1[k][frames,:,:],axis=0)
                self._rmsMap_cam1[k] = np.nanstd(dico_cam1[k][frames,:,:],axis=0)
                self._weightMap_cam1[k] = np.nansum(np.isfinite(dico_cam1[k][frames,:,:]),axis=0)
                self._masterFrame_cam2[k] = np.nanmean(dico_cam2[k][frames,:,:],axis=0)
                self._rmsMap_cam2[k] = np.nanstd(dico_cam2[k][frames,:,:],axis=0)
                self._weightMap_cam2[k] = np.nansum(np.isfinite(dico_cam2[k][frames,:,:]),axis=0)
            elif method == 'sigma_clip':
                masked_array_cam1=sigma_clip(dico_cam1[k][frames,:,:],3,1,axis=0)
                self._weightMap_cam1[k] = np.sum(masked_array_cam1.mask,axis=0).data
                self._rmsMap_cam1[k] = np.std(masked_array_cam1,axis=0).data
                self._masterFrame_cam1[k] = np.mean(masked_array_cam1,axis=0).data
                masked_array_cam2=sigma_clip(dico_cam2[k][frames,:,:],3,1,axis=0)
                self._weightMap_cam2[k] = np.sum(masked_array_cam2.mask,axis=0).data
                self._rmsMap_cam2[k] = np.std(masked_array_cam2,axis=0).data
                self._masterFrame_cam2[k] = np.mean(masked_array_cam2,axis=0).data
            else:
                raise Exception('Combination method not available: {0:s}'.format(method))
            if debug:     
                nbFrames = len(frames)
                nbDeviantFrames = 0
                deviation_cam1[k] = np.ndarray([nbFrames],dtype=np.float64)
                deviation_cam2[k] = np.ndarray([nbFrames],dtype=np.float64)
                for z_index,z in enumerate(frames):
                    medMasterCam1 = np.median(self._masterFrame_cam1[k])
                    medMasterCam2 = np.median(self._masterFrame_cam2[k])
                    medFramCam1 = np.nanmedian(dico_cam1[k][z,:,:])
                    medFramCam2 = np.nanmedian(dico_cam2[k][z,:,:])
                    deviation_cam1[k][z_index] = (medFramCam1-medMasterCam1)/medMasterCam1
                    deviation_cam2[k][z_index] = (medFramCam2-medMasterCam2)/medMasterCam2
                    if np.abs(deviation_cam1[k][z_index])>0.01:
                        print('Warning the frame {0:s} in {1:s} from camera 1 is bad: median frame value={2:6.2f}, median cube value={3:6.2f}'.format(str(z),k,deviation_cam1[k][z_index],medFramCam1))
                        nbDeviantFrames += 1
                    if np.abs(deviation_cam2[k][z_index])>0.01:
                        print('Warning the frame {0:s} in {1:s} from camera 2 is bad: median frame value={2:6.2f}, median cube value={3:6.2f}'.format(str(z),k,deviation_cam2[k][z_index],medFramCam2))
                        nbDeviantFrames += 1
        if debug:            
            if nbDeviantFrames ==0:          
                print('No deviant frames found')
            plt.figure()
            plt.clf()
            for k in self._frameTypes:
#                median_deviation = np.median(deviation_cam1[k])
#                std_deviation = np.nanstd(deviation_cam1[k])
#                isDeviant = np.abs(deviation_cam1[k]-median_deviation) > 3*std_deviation
#                nbDeviantFiles = np.sum(isDeviant)
#                if nbDeviantFiles > 0:
#                    print('Warning the file {0:s} from camera 1 contains {1:4.0f} deviant frames of indices :'.format(k,nbDeviantFiles))
#                    print(np.arange(nbFrames)[isDeviant])
#                    print(deviation_cam1[k])
                plt.plot(deviation_cam1[k],label=k)
            plt.xlabel('Frame number')
            plt.ylabel('Relative deviation to the median image')
            plt.legend(loc='lower right',frameon=False)
            plt.savefig(os.path.join(self._pathReduc,self._name+'_median_deviation_to_median_image_'+k+'_cam1.png'))
            plt.close()

            plt.figure()
            plt.clf()
            for k in self._frameTypes:
#                median_deviation = np.median(deviation_cam2[k])
#                std_deviation = np.nanstd(deviation_cam2[k])
#                isDeviant = np.abs(deviation_cam2[k]-median_deviation) > 3*std_deviation
#                nbDeviantFiles = np.sum(isDeviant)
#                if nbDeviantFiles > 0:
#                    print('Warning the file {0:s} from camera 1 contains {1:4.0f} deviant frames of indices :'.format(k,nbDeviantFiles))
#                    print(np.arange(nbFrames)[isDeviant])
                plt.plot(deviation_cam2[k],label=k)
            plt.xlabel('Frame number')
            plt.ylabel('Relative deviation to the median image')
            plt.legend(loc='lower right',frameon=False)
            plt.savefig(os.path.join(self._pathReduc,self._name+'_median_deviation_to_median_image_'+k+'_cam2.png'))
            plt.close()
            
    def getStatistics(self,value='stdev',verbose=False):
        """
        Computes a statistical parameter of the frames. By default the parameter 
        is the standard deviation. It can also be the median.
        It returns a dictionary for each camera, and prints iptionnally the 
        values of the parameters.
        Input:
            - value: a string, either 'stdev' (the default value) or 'median' to 
                    choose the statistical term to include
            - verbose: 
        Output:
            - 2 dictionnaries: each dictionnary contains the keys ['pi_odd', 'pi_even', 
            '0_even', '0_odd'] and stores the values of the statistical parameter
        """
        dico_cam1 = {}
        dico_cam2 = {}
        for k in self._frameTypes:
            if value == 'stdev':
                dico_cam1[k] = np.nanstd(self._masterFrame_cam1[k])
                dico_cam2[k] = np.nanstd(self._masterFrame_cam2[k])
            elif value == 'median':
                dico_cam1[k] = np.nanmedian(self._masterFrame_cam1[k])
                dico_cam2[k] = np.nanmedian(self._masterFrame_cam2[k])            
            else:
                raise Exception('Statistics estimator not understood: '+value)
            if verbose:
                print('Statistics: {0:s} for {1:s}'.format(value,k))
                print('Camera 1: {0:7.4e}'.format(dico_cam1[k]))
                print('Camera 2: {0:7.4e}'.format(dico_cam2[k]))
        return dico_cam1,dico_cam2
    
    def write(self,masterOnly=True):
        """
        Writes the master frame, either the master frame for each of the 
        'pi_odd', 'pi_even','0_even', '0_odd' frames or additional images with
        the RMS, weight and bad pixel frames for each of the frames.
        """
        for k in self._frameTypes:
            masterHDU_cam1 = fits.PrimaryHDU(self._masterFrame_cam1[k],header=self._header)
            masterHDU_cam1.writeto(os.path.join(self._pathReduc,self._name+'_master_'+k+'_cam1.fits'),clobber=True,output_verify='ignore')
            masterHDU_cam2 = fits.PrimaryHDU(self._masterFrame_cam2[k],header=self._header)
            masterHDU_cam2.writeto(os.path.join(self._pathReduc,self._name+'_master_'+k+'_cam2.fits'),clobber=True,output_verify='ignore')
            if not masterOnly:
                rmsHDU_cam1 = fits.PrimaryHDU(self._rmsMap_cam1[k],header=self._header)
                rmsHDU_cam1.writeto(os.path.join(self._pathReduc,self._name+'_rms_'+k+'_cam1.fits'),clobber=True,output_verify='ignore')
                badPixelHDU_cam1 = fits.PrimaryHDU(self._badPixelMap_cam1[k],header=self._header)
                badPixelHDU_cam1.writeto(os.path.join(self._pathReduc,self._name+'_badpixel_'+k+'_cam1.fits'),clobber=True,output_verify='ignore')
                weightHDU_cam1 = fits.PrimaryHDU(self._weightMap_cam1[k],header=self._header)
                weightHDU_cam1.writeto(os.path.join(self._pathReduc,self._name+'_weight_'+k+'_cam1.fits'),clobber=True,output_verify='ignore')
                rmsHDU_cam2 = fits.PrimaryHDU(self._rmsMap_cam2[k],header=self._header)
                rmsHDU_cam2.writeto(os.path.join(self._pathReduc,self._name+'_rms_'+k+'_cam2.fits'),clobber=True,output_verify='ignore')
                badPixelHDU_cam2 = fits.PrimaryHDU(self._badPixelMap_cam2[k],header=self._header)
                badPixelHDU_cam2.writeto(os.path.join(self._pathReduc,self._name+'_badpixel_'+k+'_cam2.fits'),clobber=True,output_verify='ignore')
                weightHDU_cam2 = fits.PrimaryHDU(self._weightMap_cam2[k],header=self._header)
                weightHDU_cam2.writeto(os.path.join(self._pathReduc,self._name+'_weight_'+k+'_cam2.fits'),clobber=True,output_verify='ignore')

if __name__=='__main__':
    pathRoot='/Volumes/DATA/JulienM/HD106906_ZIMPOL'
    pathRaw=os.path.join(pathRoot,'raw_calib')
    pathReduc=os.path.join(pathRoot,'calib')
    fileNames='SPHER.2015-07-21T14:42'
    masterFile=ZimpolMasterFile(pathRaw,pathReduc,fileNames,name='SPHER.2015-07-21T14:42_bias_fastpol')
    masterFile.collapseFrames(debug=True)
    masterFile.write(masterOnly=True)
    print(masterFile.getStatistics(verbose=True)[0])
#    masterFile.write(path)
    
