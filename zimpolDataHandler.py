# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:22:00 2015

@author: jmilli
"""

import numpy as np
from astropy.io import fits
import os
from dataHandler import DataHandler
from rebin import rebin3d

class ZimpolDataHandler(DataHandler):
    """This class represents a ZimpolDataHandler object. It inherits from the 
    DataHandler object, and has attributes and methods specific for Zimpol.
    Common attributes with DataHandler:
        - _pathRaw: the absolute path where the raw files are stored
        - _pathReduc: the absolute path where the reduced files
                      are stored
        - _fileNames: the list of filenames
        - _keywords: a dictionary on keywords. Each element of the dictionnary  
                     is a list of the keyword values of each file.
        - _name
    Specific attributes to ZimpolDataHandler:
        - _prescanColumns
        - _overscanColumns
        - _originalColumnNb
        - _originalRowNb
        - _keywordList
        - _cameras
        - _frameTypes
    Common methods to ZimpolDataHandler:
        - writeMetaData
        - loadFiles
        - testPath
        - getNumberFiles
        - getFileNames
        - getKeywords
        - getName
    Specific methods to ZimpolDataHandler:
        - _extractFramesFromFile
        - extractFramesFromFiles
        - _extractFramesFromCube
        - rebinColumns
        - getRowNumber
        - getColumnNumber
        - getCameras
        - getFrameTypes
        """
       
     # class variables
    _originalColumnNb=1156 # this is the total column number but do not correspond to physical pixel columns 
    _prescanColumns=25 # the 25 first columns are prescan, then come 1024/2=512 columns read by the left ADC 
    _overscanColumns=41 # then come 41 columns of overscan
    _originalRowNb=1024 # this correspond to real physical rows. Each odd row is masked
    _rowNb = _originalRowNb//2 #-1 # 510 row for each odd/even frame (there are 2 crap rows due to the phase shifting pinciple)
    _columnNb = _originalColumnNb - 2*_prescanColumns -2*_overscanColumns #1024
    _keywordList = ['HIERARCH ESO DPR TYPE','HIERARCH ESO DET NDIT', \
                    'HIERARCH ESO DET READ CURNAME','HIERARCH ESO DET DIT1', \
                    'HIERARCH ESO DPR TECH', 'HIERARCH ESO DPR CATG',\
                    'HIERARCH ESO TPL ID' , 'HIERARCH ESO INS3 OPTI2 NAME',\
                    'HIERARCH ESO INS3 OPTI5 NAME','HIERARCH ESO INS3 OPTI6 NAME',\
                    'HIERARCH ESO INS4 DROT2 POSANG','HIERARCH ESO INS3 POS6 POS',\
                    'HIERARCH ESO INS3 POS7 POS','HIERARCH ESO INS4 DROT2 MODE',\
                    'HIERARCH ESO INS4 DROT3 POSANG','HIERARCH ESO INS3 POS3 POS',\
                    'HIERARCH ESO INS3 POS4 POS','HIERARCH ESO INS3 POS6 POS',\
                    'HIERARCH ESO INS3 POS7 POS', 'HIERARCH ESO INS3 POS2 POS']
    _cameras=[1,2]
    _frameTypes = ['0_even', '0_odd','pi_even','pi_odd']                      
#    _frameTypes = ['even','odd']                      
    _indicesRealColumnsleft  = range(_prescanColumns,_originalColumnNb//2-_overscanColumns) 
    _indicesRealColumnsright = range(_originalColumnNb//2+_overscanColumns,_originalColumnNb-_prescanColumns)
    _indicesRealColumns = list(_indicesRealColumnsleft) + list(_indicesRealColumnsright)
#  _indicesDarkLeft = range(1,_prescanColumns) + range(_originalColumnNb/2 - \
#            _overscanColumns,_originalColumnNb/2) # warning the prescans are not clean, we therefore only use the overscan
    _indicesDarkLeft = range(_originalColumnNb//2 - _overscanColumns,_originalColumnNb//2) 
#    _indicesDarkRight = range(_originalColumnNb/2,_originalColumnNb/2 + \
#            _overscanColumns) + range(_originalColumnNb-_prescanColumns, \
#            _originalColumnNb-1)  # warning the prescans are not clean, we therefore only use the overscan
    _indicesDarkRight = range(_originalColumnNb//2,_originalColumnNb//2 + _overscanColumns) 
    _indicesEvenRows=range(0,_originalRowNb,2)
    _indicesOddRows=range(1,_originalRowNb,2)
    
    def __init__(self,pathRaw,pathReduc,fileNames,name='zimpol_file'):
        """
        Constructor of the class ZimpolDataHandler. It takes the same input 
        as DataHandler.
        It creates a subfolder for each polar cycle.
        Input:
            - pathRaw: the absolute path where the raw files are stored
            - pathReduc: the absolute path where the reduced files
                         are stored
            - fileNames: the list of filenames. It can be either a string 
                        with the general start of the file names, e.g. 'SPHERE_ZIMPOL_', 
                        or a list of complete filenames
            - name: optional name for the file (zimpol_file by default)
        """
        DataHandler.__init__(self,pathRaw,pathReduc,self._keywordList,fileNames,name)

    def _extractFramesFromFile(self,index,subtractDark=False):
        """
        Function that opens an individual raw frame, extracts the 0 and pi frame for each
        even and odd rows, and for each of the two cameras.
        Input:
            - index: the index of the fits files to extract
            - subtractDark: boolean: to subtract the dark or not.
        Output:
            - dico: a dictionnary containing 2 entries for each camera (1 and 2),
                    plus a third entry 'header' containing the header of the fits 
                    file.
                    Each camera entry is a dictionary containing the entries 
                     'zero_odd','zero_even', 'pi_odd' and 'pi_even' in the 
                     form of a cube of dimension [nz,512,512]. 
        """
        if isinstance(index,(list,range)):
            if len(index) > 1:
                return self.extractFramesFromFiles(self,index)
        if index>=self.getNumberFiles():
            raise Exception('Trying to extract frames from file {0:3d}:'.format(index),\
                            'index out of bound exception')
        fileName=self._fileNames[index]
        print('Extracting frame {0:3d} : {1:s}'.format(index,fileName))
        hduList = fits.open(os.path.join(self._pathRaw, fileName))

#            dither = {1:[0,0],2:[0,0]}
        shiftx_cam_1 = int(self._keywords['HIERARCH ESO INS3 POS3 POS'][index])
        shifty_cam_1 = int(self._keywords['HIERARCH ESO INS3 POS4 POS'][index]+\
                           self._keywords['HIERARCH ESO INS3 POS2 POS'][index])
        shiftx_cam_2 = int(self._keywords['HIERARCH ESO INS3 POS6 POS'][index])
        shifty_cam_2 = int(self._keywords['HIERARCH ESO INS3 POS7 POS'][index])
        print('Dithering  correction [X,Y] by [{0:d},{1:d}]'.format(shiftx_cam_1,\
              shifty_cam_1),' native pixels for cam1 and ',
              '[{0:d},{1:d}] for cam2'.format(shiftx_cam_2,shifty_cam_2))
        self.dither = {1:[shiftx_cam_1,shifty_cam_1//2],2:[shiftx_cam_2,\
                       shifty_cam_2//2]}        
        header=hduList[0].header
        dico={}
        for camera in self._cameras:
            dico[camera]=self._extractFramesFromCube(hduList[camera].data,\
                subtractDark=subtractDark,dither=self.dither[camera])
        hduList.close()
        dico['header']=header
        return dico
        
    def extractFramesFromFiles(self,indices=None,subtractDark=False):
        """
        Function identical as extractFramesFromFile but for a list of indices.
        Input:
            - indices: a list of indices (between 0 and getNumberFiles()-1). By default
                        it extracts all frames from 0 to getNumberFiles()-1
        Output: 
            - dico: a dictionnary containing 2 entries for each camera (1 and 2),
                    plus a third entry 'header' containing the header of the fits 
                    file corresponding to the last index.
                    Each camera entry is a dictionary containing the entries 
                     'zero_odd','zero_even', 'pi_odd' and 'pi_even' in the 
                     form of a cube of dimension [nz,512,512]. 
        """
        if indices==None:
            return self.extractFramesFromFiles(range(self.getNumberFiles()),\
                                               subtractDark=subtractDark)
        if not isinstance(indices,(list,range)):
            return self._extractFramesFromFile(indices)
        else:
            if len(indices) == 1:
                return self._extractFramesFromFile(indices[0],subtractDark=subtractDark)
            totalNbFrames=np.sum([self._keywords['HIERARCH ESO DET NDIT'][i] for i in indices])            
        dico={}
        # we initialize the cubes of images for each camera and each frame type
        for camera in self._cameras:
            dico[camera]={}
            for frameType in self._frameTypes:
                dico[camera][frameType] = np.ndarray([totalNbFrames//2,self._rowNb,self._columnNb])
        i=0 #index over the frames
        for index in indices:
            nbFrames = self._keywords['HIERARCH ESO DET NDIT'][index]
            dicoSingleFrame = self._extractFramesFromFile(index)
            for camera in self._cameras:
                for frameType in self._frameTypes:
                    dico[camera][frameType][range(i,i+nbFrames//2),:,:] = \
                        dicoSingleFrame[camera][frameType]
            i += nbFrames//2 # we divide by 2 because of 0 and pi frames
        dico['header'] = dicoSingleFrame['header']        
        return dico
        
    def _extractFramesFromCube(self,cube,subtractDark=False,dither=[0,0]):
        """
        Function that takes a raw cube of image and returns 4 cubes in a dictionnary
        optionnally subtracted by the dark level as estimated in the overscan and 
        prescan region.
        Input:
            - cube 
            - subtractDar: boolean for dark subtraction from the pre/overscan region
            - dither: 2-element array for shift in case dither was used
        """
        nbFrames=cube.shape[0]
#        if badframes is not None:
#            if isinstance(badframes,(list,range)):
#                list_frames = np.arange(nbFrames)
#                goodframes = [idf for idf in list_frames if idf not in badframes]
#                cube = cube[goodframes,:,:]
#                nbFrames=cube.shape[0]
#                print('Bad frames specified (0-based):',badframes)
#            else:
#                raise Exception('badframes should be specified with a list.',
#                                'Got ',badframes)
        startPhase0 = 0
        startPhasePi = 1
        if nbFrames % 2 != 0:
            raise ValueError('The number of frames is odd: {0:3d}'.format(nbFrames))
        if cube.shape[2] != self._originalColumnNb:
            raise ValueError('The number of columns is different from {0:4d} : {1:4d}'.format(self._originalColumnNb,cube.shape[2]))
        if cube.shape[1] != self._originalRowNb:
            raise ValueError('The number of rows is different from {0:4d} : {1:4d}'.format(self._originalRowNb,cube.shape[1]))
        dico={}
#        phase0Even  =cube[startPhase0 :nbFrames:2,0:cube.shape[1]-2:2,self._indicesRealColumns].astype('float64')
#        phase0Odd   =cube[startPhase0 :nbFrames:2,1:cube.shape[1]-1:2,self._indicesRealColumns].astype('float64')
#        phasePiEven =cube[startPhasePi:nbFrames:2,2:cube.shape[1]  :2,self._indicesRealColumns].astype('float64')
#        phasePiOdd  =cube[startPhasePi:nbFrames:2,1:cube.shape[1]-1:2,self._indicesRealColumns].astype('float64')
        phase0Even  =         cube[startPhase0 :nbFrames:2,0:cube.shape[1]:2,self._indicesRealColumns].astype('float64')
        phase0Odd   =         cube[startPhase0 :nbFrames:2,1:cube.shape[1]:2,self._indicesRealColumns].astype('float64')
        phasePiEven =         cube[startPhasePi:nbFrames:2,0:cube.shape[1]:2,self._indicesRealColumns].astype('float64')
        phasePiOdd  = np.roll(cube[startPhasePi:nbFrames:2,1:cube.shape[1]:2,self._indicesRealColumns].astype('float64'),1,axis=1)
        if subtractDark:
            # the dark values read from the pre/over scans is averaged spatially (rows and columns direction)
            # we keep only the temporal dimension
            dark_phase0_left   = np.median(cube[startPhase0 :nbFrames:2,:,self._indicesDarkLeft],axis=(2)) #1d sequence of dark values
            dark_phasePi_left  = np.median(cube[startPhasePi:nbFrames:2,:,self._indicesDarkLeft],axis=(2)) #1d sequence of dark values
            dark_phase0_right  = np.median(cube[startPhase0 :nbFrames:2,:,self._indicesDarkRight],axis=(2)) #1d sequence of dark values
            dark_phasePi_right = np.median(cube[startPhasePi:nbFrames:2,:,self._indicesDarkRight],axis=(2)) #1d sequence of dark values
            # The dark is the same for even and row frames
            phase0Even[:,:,0:self._columnNb//2] -= dark_phase0_left[:,0:cube.shape[1]:2,np.newaxis]
            phase0Even[:,:,self._columnNb//2: ] -= dark_phase0_right[:,0:cube.shape[1]:2,np.newaxis]
            phase0Odd[:,:,0:self._columnNb//2]  -= dark_phase0_left[:,1:cube.shape[1]:2,np.newaxis]
            phase0Odd[:,:,self._columnNb//2:]  -= dark_phase0_right[:,1:cube.shape[1]:2,np.newaxis]

            phasePiEven[:,:,0:self._columnNb//2] -= dark_phasePi_left[:,0:cube.shape[1]:2,np.newaxis]
            phasePiEven[:,:,self._columnNb//2: ] -= dark_phasePi_right[:,0:cube.shape[1]:2,np.newaxis]
            phasePiOdd[:,:,0:self._columnNb//2]  -= dark_phasePi_left[:,1:cube.shape[1]:2,np.newaxis]
            phasePiOdd[:,:,self._columnNb//2: ]  -= dark_phasePi_right[:,1:cube.shape[1]:2,np.newaxis]
        if dither!=[0,0]:
            if len(dither)!=2:
                ValueError('The dither must be a two-elements array [shiftx,shifty]')
            phase0Even = np.roll(phase0Even,dither[1],axis=1) # y axis (INS3 POS4 POS/2 for cam1 and INS3 POS7 POS/2 for cam2)
            phase0Even = np.roll(phase0Even,dither[0],axis=2) # x axis (INS3 POS3 POS for cam1 and INS3 POS6 POS for cam2)
            phase0Odd = np.roll(phase0Odd,dither[1],axis=1) # y axis (INS3 POS4 POS/2 for cam1 and INS3 POS7 POS/2 for cam2)
            phase0Odd = np.roll(phase0Odd,dither[0],axis=2) # x axis (INS3 POS3 POS for cam1 and INS3 POS6 POS for cam2)
            phasePiEven = np.roll(phasePiEven,dither[1],axis=1) # y axis (INS3 POS4 POS/2 for cam1 and INS3 POS7 POS/2 for cam2)
            phasePiEven = np.roll(phasePiEven,dither[0],axis=2) # x axis (INS3 POS3 POS for cam1 and INS3 POS6 POS for cam2)
            phasePiOdd = np.roll(phasePiOdd,dither[1],axis=1) # y axis (INS3 POS4 POS/2 for cam1 and INS3 POS7 POS/2 for cam2)
            phasePiOdd = np.roll(phasePiOdd,dither[0],axis=2) # x axis (INS3 POS3 POS for cam1 and INS3 POS6 POS for cam2)
#            fits.writeto('Users/jmilli/desktop/tmp_after_shift.fits',np.median(phase0Even,axis=3),overwrite=True)


        ## For the odd frame, we shift one column up the image because of row 10 is to
        ## be subtracted with row 9 in the 0 frame, row 10 is to be subtracted to row 11
        ## in the pi frame. 
#        phasePiOdd=np.roll(cube[startPhasePi:nbFrames:2,1:cube.shape[1]:2,self._indicesRealColumns],1,axis=1)
        dico[self._frameTypes[0]]=phase0Even #['0_even', '0_odd','pi_even','pi_odd']  
        dico[self._frameTypes[1]]=phase0Odd
        dico[self._frameTypes[2]]=phasePiEven
        dico[self._frameTypes[3]]=phasePiOdd
        return dico
        
    def rebinColumns(self,cube):
        """
        Rebins a Zimpol cube to obtain squared pixels. It reduces the number of 
        columns by a factor 2 in order to get squared pixels of 7mas/px
        """
        return rebin3d(cube,(cube.shape[1],cube.shape[2]//2))
                
    def getRowNumber(self):
        """
        Returns the class global attribute _rowNb (hard-coded to 1024)
        """
        return self._rowNb

    def getColumnNumber(self):
        """
        Returns the class global attribute _columnNb (hard-coded to 1156)
        """
        return self._columnNb
        
    def getCameras(self):   
        """
        Returns the class global attribute _cameras (hard-coded to [0,1])
        """
        return self._cameras

    def getFrameTypes(self):
        """
        Returns the class global attribute _cameras (hard-coded to ['0_even', 
        '0_odd','pi_even','pi_odd'])
        """        
        return self._frameTypes
        
    def getTotalNumberFrames(self):
        """
        Returns the total number of frames summing up all files
        """
        i=0 #index over the frames
        for index in range(self.getNumberFiles()):
            i += self._keywords['HIERARCH ESO DET NDIT'][index]
        return i        

    def _get_id_from_dpr_catg(self,frameType='all'):
        """
        Internal function that returns a list with the indices of the requested frames
        (either "all", "SCIENCE" or "CALIB")
        """
        if frameType == 'all':
            idFrames = range(self.getNumberFiles())
        else:
            dpr_catg = self._keywords['HIERARCH ESO DPR CATG']
            idFrames = []
            for index,dpr in enumerate(dpr_catg):
                if dpr == frameType:
                    idFrames.append(index)
#        if len(idFrames) == 0:
#            raise TypeError('The frameType keyword must be "all", "SCIENCE" or "CALIB"')
        return idFrames

        
if __name__=='__main__':
    pathRoot='/Volumes/MILOU_1TB_2/HR4796_zimpol_GTO'
    pathRaw=os.path.join(pathRoot,'raw_science/FastPolarimetry_000')
    pathReduc=os.path.join(pathRoot,'test')
    fileNames='SPHER.2016-05-25T01:14:12.495.fits'
    zimpolDataHandler = ZimpolDataHandler(pathRaw,pathReduc,fileNames,name='test')
    zimpolDataHandler.getTotalNumberFrames()
    dico = zimpolDataHandler._extractFramesFromFile(0,\
                    subtractDark=False)
    
#    pathRoot='/Volumes/DATA/JulienM/HD106906_ZIMPOL'
#    pathRaw=os.path.join(pathRoot,'raw_calib')
#    pathReduc=os.path.join(pathRoot,'calib')
#    fileNames='SPHER.2015-07-21T14:42'
#    bias = ZimpolDataHandler(pathRoot,pathRaw,pathReduc,fileNames)
#    #dico = biasIO._extractFramesFromFile(0)
#    dico = bias._extractFramesFromFile(0)
#    import pyds9
#    ds9=pyds9.DS9()
#    ds9.set_np2arr(dico[1]['pi_odd'][0,:,:])