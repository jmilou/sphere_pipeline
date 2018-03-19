#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:20:39 2017

@author: jmilli
"""
import numpy as np

class PolarimetricCycle():
    """
    Polarimetric cycle object that is instantiated with 4 ZimpolScienceCube objects 
    corresponding to the Stokes +Q, -Q, +U and -U 
    Attributes of PolarimetricCycle:
        - recenter: way to recenter the frames between 'default' (no centering),
                    'gaussianFit' and 'correlation'
        - beamShift: boolean (if True corrects the beam shift)
        - 4 ZimpolScienceCube objects for +Q, -Q, +U and -U 
        - masterBias: a masterBias object to correct the I image from the bias.
    
    It creates final Stokes I, Q and U (a 2-element list for cam1 and cam2)
    """
    
    def __init__(self,plusQ,minusQ,plusU,minusU,beamShift=False,center=None):
        """
        """
        self._plusQ = plusQ
        self._minusQ = minusQ
        self._plusU = plusU
        self._minusU = minusU
        self._guess_center_xy = center
        self.beamShift=beamShift
        self.computeStokes()
        return
    
    def computeStokes(self):
        """
        """
        self._plusQ.collapseFrames()
        self._plusQ.computeStokes(beamShift=self.beamShift)
        self._minusQ.collapseFrames()
        self._minusQ.computeStokes(beamShift=self.beamShift)
        self._plusU.collapseFrames()
        self._plusU.computeStokes(beamShift=self.beamShift)
        self._minusU.collapseFrames()
        self._minusU.computeStokes(beamShift=self.beamShift)
        self._I = []
        self._Q = []
        self._U = []
        for icam,cam in enumerate(self._plusQ.getCameras()):
            I = (self._plusQ._even_plus_odd[icam]+self._minusQ._even_plus_odd[icam]+\
                 self._plusU._even_plus_odd[icam]+self._minusU._even_plus_odd[icam])/4
            self._I.append(I)
            Q = (self._plusQ._even_minus_odd[icam] - self._minusQ._even_minus_odd[icam])/2.
            self._Q.append(Q)
            U = (self._plusU._even_minus_odd[icam] - self._minusU._even_minus_odd[icam])/2.
            self._U.append(U)
        return

    def print_beam_shift_statistics(self):
        """
        Displays and saves the statistics of the beam shift to be re-used later
        """
        description = ['+Q','-Q','+U','-U']
        mean_beamshift = np.ndarray((4,2,2,2))
        std_beamshift = np.ndarray((4,2,2,2))
        #(3 dimensions: camera,phase,and direction)
        for i,scienceCube in enumerate([self._plusQ,self._minusQ,self._plusU,self._minusU]):
            print('Beam shift statistics for {0:s}'.format(description))
            mean_beamshift_Stokes,std_beamshift_Stokes = scienceCube._print_beam_shift_statistics()
            mean_beamshift[i,:,:,:] = mean_beamshift_Stokes
            std_beamshift[i,:,:,:] = std_beamshift_Stokes
        return mean_beamshift,std_beamshift
    
    def _writeAllFrames(self):
        self._plusQ.writeCube(allFrames=True)
        self._minusQ.writeCube(allFrames=True)
        self._plusU.writeCube(allFrames=True)
        self._minusU.writeCube(allFrames=True)                