# -*- coding: utf-8 -*-
"""
Created on Aug 31 15:34:37 2015

@author: jmilli
"""

import numpy as np
from astropy.io import ascii,fits
import os
import glob

class DataHandler(object):
    """This class is a parent class for the calibration or science file classes.
    Attributes:
        - _pathRoot: the absolute path where the reduction is performed
        - _pathRaw: the absolute path where the raw files are stored
        - _pathReduc: the absolute path where the reduced files
                      are stored
        - _fileNames: the list of filenames. It can be either a string 
                      with the general start of the 
                      file names, e.g. 'SPHERE_ZIMPOL_', or a list of complete filenames
        - _keywords: a dictionary on keywords. Each element of the dictionnary  
                     is a list of the keyword values of each file.
        - _name: a name for this assocation of files
    Methods:
        - writeMetaData
        - loadFiles
        - testPath
        - getNumberFiles
        - getFileNames
        - getKeywords
        - getName
    """

    def __init__(self,pathRaw,pathReduc,keywordList,fileNames,name='default'):
        """
        Constructor of the class
        Input:
            - pathRaw: the absolute path where the raw files are stored
            - pathReduc: the absolute path where the reduced files
                         are stored
            - keywordList: a list of keywords (strings).
            - fileNames: the list of filenames (string). It can be either a string 
              with the general start of the file names, e.g. 'SPHERE_ZIMPOL_', 
              or a list of complete filenames.
            - firstHeader: the full header of the first file
        """
        self._pathRaw = pathRaw
        self.testPath(self._pathRaw)
        self._pathReduc = pathReduc
        self.testPath(self._pathReduc)
        self._fileNames = []
        self._name = name
        self._keywords = {}
        for keywordName in keywordList:
            self._keywords[keywordName]=[]
        self.loadFiles(self._pathRaw,fileNames)
        nbFiles = self.getNumberFiles()
        print('There are {0:3d} raw files'.format(nbFiles))

    def writeMetaData(self,fileName='keyword_list.csv'):
        """
        Function that writes an ascii file in pathReduc with the value of 
        each keyword  
        Input:
            - fileName: the name of the ascii file (default: keyword_list.txt)
        Output:
            None
        """
        ascii.write([self._keywords[key] for key in self._keywords.keys()],
                         os.path.join(self._pathReduc,fileName),names=self._keywords.keys(),format='csv')
        print('Wrote the file {0:s}'.format(os.path.join(self._pathReduc,fileName)))
            
    def loadFiles(self,dir,inputFiles,verbose=False):
        """
        Function that populates the _fileNames attribute of the CalibrationFile 
        object and build the list of keywords. 
        Input:
            - inputFiles: it can be either a string with the general start of the 
            file names, e.g. 'SPHERE_ZIMPOL_', or a list of complete filenames
        """
        if type(inputFiles) == list or type(inputFiles) == np.ndarray:
            for inputFile in inputFiles:
                if os.path.isfile(os.path.join(self._pathRaw,inputFile)):
                    self._fileNames.append(inputFile)
        elif type(inputFiles) == str or type(inputFiles) == np.string_:
            for fileName in sorted(glob.glob(os.path.join(self._pathRaw,inputFiles))):
                if fileName.endswith('.fits'):
                    self._fileNames.append(fileName)
#            for root, dirs, f in os.walk(self._pathRaw): 
#                for fileName in f:
#                    if fileName.startswith(inputFiles) and fileName.endswith('.fits'):
#                        self._fileNames.append(fileName)
        else:
            raise Exception('The argument of loadFiles must be a string or a list of strings')
        if len(self._fileNames)<1:
            raise Exception('No files found')            
        for fileName in self._fileNames:
            print('Reading file {0:s}'.format(fileName))
            hdulist = fits.open(os.path.join(self._pathRaw, fileName))
            for keyword in self._keywords.keys():
                keywordValue=hdulist[0].header[keyword]
                self._keywords[keyword].append(keywordValue)
                if verbose:
                    print('{0:s} : {1:s}'.format(keyword,str(keywordValue)))
            hdulist.close()
        self.firstHeader = fits.getheader(os.path.join(self._pathRaw, self._fileNames[0]))

    def testPath(self,path):
        """
        Function that test if a path exists and if not creates it.
        Input:
            - path: absolute path name of the path to test (string)
        """
        if not os.path.exists(path):
            print('The input directory {0:s} does not exist and was created.'.format(path))
            os.makedirs(path)

    def getNumberFiles(self):
        """
        Returns the number of frames of the object fitsFileIO
        """
        return len(self._fileNames)

    def getKeywords(self):
        """
        Returns the class attributes _keywords
        """
        return self._keywords
    
    def getFileNames(self):
        """
        Returns the class attributes _fileNames
        """
        return self._fileNames  
        
    def getName(self):
        """
        Returns the name
        """
        return self._name

if __name__=='__main__':
    pathRoot='/Volumes/DATA/JulienM/HD106906_ZIMPOL'
    pathRaw=os.path.join(pathRoot,'raw_calib')
    pathReduc=os.path.join(pathRoot,'calib')
    fileNames='SPHER.2015-07-21T14:42'
    keywordList = ['HIERARCH ESO DPR TYPE','HIERARCH ESO DET NDIT', \
    'HIERARCH ESO DET READ CURNAME','HIERARCH ESO DET DIT1','HIERARCH ESO DPR TECH']
    bias = DataHandler(pathRaw,pathReduc,keywordList,fileNames)
    bias.writeMetaData('bias_metadata.txt')