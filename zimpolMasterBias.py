# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:22:00 2015

@author: jmilli
"""

from zimpolMasterFile import ZimpolMasterFile

class ZimpolMasterBias(ZimpolMasterFile):
    """
    ZimpolMasterBias object that inherits from ZimpolMasterFile. This object
    is used to represent a bias image.
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
        The latter 8 attributes are dictionnaries containing the following entries:
        'pi_odd', 'pi_even', '0_even', and '0_odd'.
    Specific attributes to ZimpolMasterBias:None
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
        - collapseFrames
    Specific methods to ZimpolMasterBias: None
    """
    def __init__(self,pathRaw,pathReduc,fileNames,badPixelMap=None,name='zimpol_master_bias'):
        """
        Constructor of the ZimpolMasterBias object. It instantiates the master frame
        with nan values. It takes the same input 
        as ZimpolDataHandler, plus an optional bad pixel map and name.
        Input:
            - pathRaw: the absolute path where the raw files are stored
            - pathReduc: the absolute path where the reduced files
                         are stored
            - fileNames: the list of filenames. It can be either a string 
                        with the general start of the file names, e.g. 'SPHERE_ZIMPOL_', 
                        or a list of complete filenames
            - badPixelMap: a map of bad pixel (bad=1, good=0) (optional) 
            - name: name of the bias frame (by default zimpol_master_bias)
        """
        ZimpolMasterFile.__init__(self,pathRaw,pathReduc,fileNames,badPixelMap,name)
        
if __name__=='__main__':
#    path='/Volumes/DATA/JulienM/HD106906_ZIMPOL/test'
#    masterBias=ZimpolMasterBias('test')
#    masterBias.write(path)
    import os
    pathRoot='/Volumes/DATA/JulienM/HD106906_ZIMPOL'
    pathRaw=os.path.join(pathRoot,'raw_calib')
    pathReduc=os.path.join(pathRoot,'calib')
    fileNames='SPHER.2015-07-21T14:42'    
    masterBias = ZimpolMasterBias(pathRaw,pathReduc,fileNames)
    masterBias.collapseFrames()
    masterBias.write(masterOnly=True)
    print(masterBias.getStatistics(verbose=True)[0])
    