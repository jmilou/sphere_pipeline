# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:22:00 2015

@author: jmilli
"""
from zimpolMasterFile import ZimpolMasterFile
from zimpolMasterBias import ZimpolMasterBias
from astropy.stats import sigma_clip
import numpy as np

class ZimpolMasterFlat(ZimpolMasterFile):
    """
    This is a master flat for zimpol.
    Common attributes with ZimpolMasterFile:
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
    Specific attributes to ZimpolMasterBias:
        - _masterBias : an object ZimpolMasterBias to be subtracted to the flat.
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
        - buildMasterFrame
        - write
    Specific methods to ZimpolMasterBias:
        - buildMasterFrame
    """
    
    def __init__(self,pathRaw,pathReduc,fileNames,badPixelMap=None,\
                 name='zimpol_master_flat',masterBias=None):
        """
        Constructor of the class ZimpolMasterFlat. It inherits form ZimpolMasterFile
        but takes an additional input for the masterDark to be subtracted to the flat
        """
        ZimpolMasterFile.__init__(self,pathRaw,pathReduc,fileNames,\
                                  badPixelMap,name)
        self._masterBias=masterBias
        
        
#                if cosmetic:
##                cube_tmp=np.ndarray(dico[k].shape)
#                for z in range(0,dico[k].shape[0]):
#                    dico[k][z,:,:] = correctBadPixelInFrame(dico[k][z,:,:],\
#                        self._badPixelMap,size=5, clip=3.)

    def _normalizeMasterFrame(self,normalization='median'):
        """
        Normalize the master frame.
        Input:
            - normalization how to normalize the master frame. By default, normalizes by 
             the median of the frame. Accepts also 'sigma_clip' (sigma clipping 
             of sigma=3 with 1 iteration) or 'mean'
        Output: nothing
        """
        for k in self._frameTypes:
            if normalization == 'median':
                normalizationValueCam1 = float(np.nanmedian(self._masterFrame_cam1[k]))
                self._masterFrame_cam1[k] /= normalizationValueCam1
                normalizationValueCam2 = float(np.nanmedian(self._masterFrame_cam2[k]))
                self._masterFrame_cam2[k] /= normalizationValueCam2
            elif normalization == 'mean':
                normalizationValueCam1 = float(np.nanmean(self._masterFrame_cam1[k]))
                self._masterFrame_cam1[k] /= normalizationValueCam1
                normalizationValueCam2 = float(np.nanmean(self._masterFrame_cam2[k]))
                self._masterFrame_cam2[k] /= normalizationValueCam2
            elif normalization == 'sigma_clip':
                masked_array_cam1=sigma_clip(self._masterFrame_cam1[k],3,1)
                normalizationValueCam1 = float(np.mean(masked_array_cam1.data[~masked_array_cam1.mask]))
                self._masterFrame_cam1[k] /= normalizationValueCam1
                masked_array_cam2=sigma_clip(self._masterFrame_cam2[k],3,1)
                normalizationValueCam2 = float(np.mean(masked_array_cam2.data[~masked_array_cam2.mask]))
                self._masterFrame_cam2[k] /= normalizationValueCam2
            else:
                raise Exception('Normalization method not available: {0:s}'.format(normalization))
            print('Normalization factor for the flat {0:s} camera 1: {1:8.2f}'.format(k,normalizationValueCam1))
            print('Normalization factor for the flat {0:s} camera 2: {1:8.2f}'.format(k,normalizationValueCam2))
            threshold = np.ndarray([self._masterFrame_cam1[k].shape[0],self._masterFrame_cam1[k].shape[1]])
            threshold.fill(1e-2)
            self._masterFrame_cam1[k][self._masterFrame_cam1[k] < threshold] = 1.
            self._masterFrame_cam2[k][self._masterFrame_cam2[k] < threshold] = 1.            
            
    def buildMasterFrame(self,method='median',normalization='median',debug=False,frames=None):
        """
        Populates the master frame. 
        Input:
            - dico: a dictionnary containing the keys ['pi_odd', 'pi_even', 
            '0_even', '0_odd']. Each entry of the dico is a cube of images.
            The cube can contain nan values, in this case, these pixels are excluded
            - method: the method to build the master frame. By default, uses a 
             median. Accepts 'sigma_clip' (sigma clipping of sigma 3 with 1 iteration)
             or 'mean'
            - normalization how to normalize the master frame. By default, normalizes by 
             the median of the frame. Accepts also 'sigma_clip' (sigma clipping 
             of sigma=3 with 1 iteration) or 'mean'

        Optional input:
             - debug: booleean (False by default). In order to check for deviant 
                      frames. Can be usefulf because the first frame is sometimes with a lower flux.
             - frames:  list of frames to collapse (by defalut, it uses all frames, e.g.
                     range(0,NDIT/2). If the first frame is bad, use range(1,NDIT/2))
        Output
        """
        self.collapseFrames(method,debug=debug,frames=frames)
        if self._masterBias != None:
            for k in self._frameTypes:
                self._masterFrame_cam1[k] -= self._masterBias._masterFrame_cam1[k]
                self._masterFrame_cam2[k] -= self._masterBias._masterFrame_cam2[k]
                self._rmsMap_cam1[k] = np.sqrt(self._rmsMap_cam1[k]**2 + \
                                               self._masterBias._rmsMap_cam1[k]**2)            
                self._rmsMap_cam2[k] = np.sqrt(self._rmsMap_cam2[k]**2 + \
                                               self._masterBias._rmsMap_cam2[k]**2)            
        self._normalizeMasterFrame(normalization=normalization)
        
if __name__=='__main__':
    pathRoot='/Volumes/DATA/JulienM/HD106906_ZIMPOL'
    import os
    pathRawCalib=os.path.join(pathRoot,'raw_calib')
    pathReducCalib=os.path.join(pathRoot,'calib')
    fileNameBias='SPHER.2015-07-21T14:42'
    masterBias=ZimpolMasterBias(pathRawCalib,pathReducCalib,fileNameBias,name='bias')
    masterBias.collapseFrames()
#    masterBias._masterFrame_cam1['0_odd'].shape

    fileNameFlat='flat1_slowpol'
    masterFlat=ZimpolMasterFlat(pathRoot,pathRawCalib,pathReducCalib,fileNameFlat,
                                name='flat',masterBias=masterBias)
    masterFlat.buildMasterFrame(debug=True)
    masterFlat.write(masterOnly=True)
    
    
