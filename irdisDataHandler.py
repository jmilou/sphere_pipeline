# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:22:00 2015
@author: jmilli


Updates:
2017-01-10: JM modifed write_master_cube and replaced
    centerxy = [self._columnNb/2,self._rowNb/2]
    by 
    centerxy = [522,476]
2017-01-10: JM modifed write_master_cube and added print('Reading {0:s}'.format(fileName))
2017-01-29: added centerxy as a class variable to be able to access it
2017-06-01: modified write_master_cube to return also the derotation angles including 
            the pupil offset and true north.
            added the get_region fonction for visualization purposes
2017-11-01: unwraped the parallactic angle and the wind direction 
        with np.rad2deg(np.unwrap(np.deg2rad(parang)))
        replaced size/2 by size//2
"""

from dataHandler import DataHandler
import angles as a
import sphere_utilities as sph
from image_tools import distance_array
import numpy as np
import os,sys
from astropy.io import ascii,fits
#from astroquery.simbad import Simbad
from astropy import coordinates as coords
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import ICRS, Galactic, FK4, FK5,EarthLocation
#from astropy.stats import sigma_clip
#import matplotlib.pyplot as plt
import vip 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import glob
import pandas as pd
from plot_sparta_data import plot_sparta_data
from query_eso_archive import interpolate_date,query_simbad
import pdb
from itertools import repeat

path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sphere_data')

class IrdisDataHandler(DataHandler):
    """This class represents an irdis file list object. It inherits from the 
    DataHandler object, and has attributes and methods specific to Irdis.
    Common attributes with DataHandler:
        - _pathRoot: the absolute path where the reduction is performed
        - _pathRaw: the absolute path where the raw files are stored
        - _pathReduc: the absolute path where the reduced files
                      are stored
        - _fileNames: the list of filenames. It can be either a string 
                      with the general start of the 
                      file names, e.g. 'SPHERE_ZIMPOL_', or a list of complete filenames
        - _keywords: a dictionary on keywords. Each element of the dictionnary  
                     is a list of the keyword values of each file.
        - _name
        - firstHeader
    Specific attributes to IrdisDataHandler:
        - pixel_scale
        - _keywordList
        - _latitude
        - _longitude
        - _rowNb (1024)
        - _columnNb (1024)
        - date : the date of observation (datetime object)
        - star: the name of the target star (string)
        - _keywordsExtra
    Common methods to IrdisDataHandler:
        - writeMetaData
        - loadFiles
        - testPath
        - getNumberFiles
        - getFileNames
        - getKeywords
        - getName
    Specific methods to IrdisDataHandler:
        """
       
     # class variables
    _rowNb = 1024 
    _columnNb = 1024 
#    _centerxy = np.array([[473,519],[476,508]]) # center of the coronagraphic mask (0-based index)
    _centerxy = np.array([[477,522],[481,511]]) # center of the coronagraphic mask (0-based index)
    _keywordList = ['HIERARCH ESO DPR TYPE','HIERARCH ESO DET NDIT', \
                    'HIERARCH ESO OBS TARG NAME','HIERARCH ESO DET SEQ1 DIT', \
                    'HIERARCH ESO DPR TECH', 'HIERARCH ESO DPR CATG',\
                    'HIERARCH ESO TPL ID' , 'HIERARCH ESO INS1 FILT NAME',\
                    'HIERARCH ESO INS1 OPTI2 NAME','HIERARCH ESO INS4 OPTI10 NAME',\
                    'HIERARCH ESO INS4 DROT2 POSANG','HIERARCH ESO INS4 OPTI11 NAME',\
                    'HIERARCH ESO INS4 FILT2 ID','HIERARCH ESO INS4 DROT2 BEGIN',\
                    'HIERARCH ESO INS4 DROT2 END','HIERARCH ESO TEL PARANG START',\
                    'HIERARCH ESO TEL PARANG END','HIERARCH ESO TEL AMBI FWHM START',\
                    'HIERARCH ESO TEL AMBI FWHM END','HIERARCH ESO TEL AMBI TAU0',\
                    'HIERARCH ESO TEL AMBI TEMP','HIERARCH ESO TEL AMBI WINDDIR',\
                    'HIERARCH ESO TEL AMBI WINDSP','HIERARCH ESO TEL GEOLAT',
                    'HIERARCH ESO TEL GEOLON','HIERARCH ESO TEL IA FWHM ',\
                    'HIERARCH ESO TEL IA FWHMLIN','HIERARCH ESO TEL TARG ALPHA',\
                    'HIERARCH ESO TEL TARG DELTA','HIERARCH ESO TEL TARG PMA',\
                    'HIERARCH ESO TEL TARG PMD','HIERARCH ESO TEL AIRM START',\
                    'HIERARCH ESO TEL AIRM END','HIERARCH ESO INS1 DITH POSX',\
                    'HIERARCH ESO INS1 DITH POSY','RA','DEC','DATE','DATE-OBS',\
                    'HIERARCH ESO INS4 DROT2 RA','HIERARCH ESO INS4 DROT2 DEC',\
                    'HIERARCH ESO TEL ALT','HIERARCH ESO TEL GEOELEV',\
                    'HIERARCH ESO INS COMB IFLT']
    _files_to_read = ['sparta_atmospheric_params*.csv','mass_dimm*.csv','dimm*.csv',\
                         'old_dimm*.csv','sparta_IR_DTTS*.csv','sparta_visible_WFS*.csv',\
                         'slodar*.csv','sphere_ambi*csv','asm*csv','ecmwf*csv']
    
    
    def __init__(self,pathRaw,pathReduc,fileNames,name='irdis_file',coordOrigin='derot',plot=True):
        """
        Constructor of the class IrdisDataHandler. It takes the same input 
        as DataHandler.
        Input:
            - pathRoot: the absolute path where the reduction is performed
            - pathRaw: the relative path (from pathRoot) where the raw files are stored
            - pathReduc: the relative path (from pathRoot) where the reduced files
                         are stored
            - fileNames: the list of filenames. It can be either a string 
                        with the general start of the file names, e.g. 'SPHERE_ZIMPOL_', 
                        or a list of complete filenames
            - name: optional name for the file (zimpol_file by default)
            - coordOrigin: string specifying the origin of the target coordinates 
                            (used for the parallactic angle calculation). Options are:
                            - 'pointing': to keywords RA and DEC
                            - 'simbad': to use OBS.TARG.NAME and retrieve the coords from Simbdad
                                        precess them to the date of obs and use the 
                                        TEL.TARG.PMA/PMD for proper motion
                            - 'target': to use the TEL.TARG.ALPHA/DEC coordinates
                                        entered by the user in the OB, precess them to 
                                        the date of observations and use the 
                                        TEL.TARG.PMA/PMD for proper motion
                            - 'derot': to use the keywords INS4.DROT2.RA/DEC used by 
                                        the CPU to compute the derotator angle.
            - plot: boolean. True to see the plot of the true north vs MJD. 
        """
        DataHandler.__init__(self,pathRaw,pathReduc,self._keywordList,fileNames,name)
        self.pixel_scale = 0.01225
        self.pupil_offset =  -135.99
        self.set_common_parameters(coordOrigin=coordOrigin)
        self.compute_properties(plot=plot)
        self.theoretical_fwhm = self.get_theoretical_fwhm()
        self.get_true_north(interpolate=False,plot=plot)
        self.set_wind()
        self._keywordsExtra = {}
        
    def getTotalNumberFrames(self,frameType='all'):
        """
        Returns the total number of frames summing up all files
        Input:
            -frameType: 'all' if all files from the file list are to be processed,
                or 'O' (OBJECT) (resp. 'F' (FLUX), 'C' (CENTER)) if only the
                object (resp. center, flux) files are to be processed.
        """
        i=0 #index over the frames
        idFrames = self._get_id_frames(frameType=frameType)
        for index in idFrames:
                i += self._keywords['HIERARCH ESO DET NDIT'][index]
        return i                
    
    def set_common_parameters(self,coordOrigin='pointing'):
        """
        It initializes the class properties star, latitude, longitude, date and computes
        the current star location on sky at the epoch of observation in order to compute
        the hour angle and parallactic angle for each individual dit. 
        It retrieves the pointing from the J2000 TEL.TARG.ALPHA/DEC keywords (unless 
        useStarCoord is set to True, and in this case it uses also the proper motion).
        A pending question remains wether this coordinate is corrected from the 
        refraction at the wavelength of observation (we would like to).
        Input:
            - coordOrigin: string specifying the origin of the target coordinates 
                            (used for the parallactic angle calculation). Options are:
                            - 'pointing': to keywords RA and DEC
                            - 'simbad': to use OBS.TARG.NAME and retrieve the coords from Simbdad
                                        precess them to the date of obs and use the 
                                        TEL.TARG.PMA/PMD for proper motion
                            - 'target': to use the TEL.TARG.ALPHA/DEC coordinates
                                        entered by the user in the OB, precess them to 
                                        the date of observations and use the 
                                        TEL.TARG.PMA/PMD for proper motion
                            - 'derot': to use the keywords INS4.DROT2.RA/DEC used by 
                                        the CPU to compute the derotator angle.        
        """
        self.star = self._keywords['HIERARCH ESO OBS TARG NAME'][0]
        # The real coordinates of UT3 are different from the ESO website...
#        self._latitude  = -(24+37/60.+30.300/3600)*u.
#        self._longitude = -(70+24/60.+9.896/3600)*u.degree
#        self._altitude  = 2635.43*u.meter
#       Les coordonnées GEOLAT,GEOLON des telescopes UT et AT sont disponible dans https://jira.eso.org/browse/VLTSW-6644
#        Les coordonnées GEOLAT,GEOLON de la reference Paranal UVW=0 sont disponibles dans https://jira.eso.org/browse/VLTSW-7059
#        La hauteur GEOELEV (~2669m) de la reference Paranal UVW=0 est disponible dans https://jira.eso.org/browse/VLTSW-7608
#        A noter que la hauteur dans le repère UVW des UTs et ATs est: 13.54 m and 4.54 m respectivement, à ajouter au GEOELEV précédent pour obtenir le GEOELEV d’un UT ou d’un AT.
        self._latitude = self._keywords['HIERARCH ESO TEL GEOLAT'][0]*u.degree
        self._longitude = self._keywords['HIERARCH ESO TEL GEOLON'][0]*u.degree  
        self._altitude = self._keywords['HIERARCH ESO TEL GEOELEV'][0]*u.meter 
        self.pma = self._keywords['HIERARCH ESO TEL TARG PMA'][0]
        self.pmd = self._keywords['HIERARCH ESO TEL TARG PMD'][0]   
        self.date_start = Time(self._keywords['DATE-OBS'][0],location=(self._longitude, self._latitude, self._altitude))
        self.date_end = Time(self._keywords['DATE'][-1],location=(self._longitude, self._latitude, self._altitude))
        fk5_timeOfObservation = coords.FK5(equinox=self.date_start.jyear_str)
        print(' ')

        # Simbad
        pointing_alpha = self._keywords['RA'][0]*u.degree
        pointing_delta = self._keywords['DEC'][0]*u.degree
        J2000_pointing_coords = coords.SkyCoord(pointing_alpha,pointing_delta,frame='fk5')
        self.simbad_dico = query_simbad(self.date_start,J2000_pointing_coords,name=self.star)
        if self.simbad_dico is not None:#'03h32m55.84496s -09d27m2.7312s', frame=ICRS)
            pd_simbad = pd.DataFrame(self.simbad_dico,index=[0])
            pd_simbad.to_csv(os.path.join(self._pathReduc,'{0:s}_simbad_info.csv'.format(self._name)))
            J2000_simbad_ra  = '{}h{}m{}s'.format(*self.simbad_dico['simbad_RA_ICRS'].split(' '))
            J2000_simbad_dec = '{}d{}m{}s'.format(*self.simbad_dico['simbad_DEC_ICRS'].split(' '))
            J2000_simbad_coords = coords.SkyCoord(ra=J2000_simbad_ra,dec=J2000_simbad_dec,frame=ICRS)
            J2000_current_epoch_simbad_ra  = '{}h{}m{}s'.format(*self.simbad_dico['simbad_RA_current'].split(' '))
            J2000_current_epoch_simbad_dec = '{}d{}m{}s'.format(*self.simbad_dico['simbad_DEC_current'].split(' '))
            J2000_current_epoch_simbad_coords = coords.SkyCoord(\
                ra=J2000_current_epoch_simbad_ra,\
                dec=J2000_current_epoch_simbad_dec,frame=ICRS)
            print('ICRS (ep=J2000.00 eq=2000.00) Simbad {0:s}'.format(
                  J2000_simbad_coords.to_string('hmsdms'),\
                  self.simbad_dico['simbad_DEC_ICRS']))
            print('ICRS (ep=J{0:7.2f} eq=2000.00) Simbad {1:s}'.format(self.date_start.jyear,\
                  J2000_current_epoch_simbad_coords.to_string('hmsdms')))

        # OB coordinates
        J2000_target_alpha_ob = a.convert_keyword_coord(self._keywords['HIERARCH ESO TEL TARG ALPHA'][0])
        J2000_target_delta_ob = a.convert_keyword_coord(self._keywords['HIERARCH ESO TEL TARG DELTA'][0])
        J2000_target_coords = coords.SkyCoord(J2000_target_alpha_ob,J2000_target_delta_ob,\
            frame='icrs',unit=(u.hourangle, u.deg))
        print('FK5  (ep=J2000.00 eq=2000.00) TEL.TARG.ALPHA/DELTA {1:s}'.format(\
              self.date_start.jyear,J2000_target_coords.to_string('hmsdms')))                   
        J2000_current_epoch_target_coords = coords.SkyCoord(\
            ra=J2000_target_coords.ra+self.pma*u.arcsec/np.cos(J2000_target_coords.dec)*(self.date_start.jyear-2000),
            dec=J2000_target_coords.dec+self.pmd*u.arcsec*(self.date_start.jyear-2000),frame=ICRS)
        print('FK5  (ep=J{0:7.2f} eq=2000.00) TEL.TARG.ALPHA/DELTA {1:s}'.format(\
              self.date_start.jyear,J2000_current_epoch_target_coords.to_string('hmsdms')))                   

        #Pointing
        print('FK5  (ep=J{0:7.2f} eq=2000.00) pointing (RA/DEC) {1:s}'.format(\
              self.date_start.jyear,J2000_pointing_coords.to_string('hmsdms')))
        Jcurrent_pointing_coords = J2000_pointing_coords.transform_to(fk5_timeOfObservation)
        print('FK5  (ep=J{0:7.2f} eq={0:7.2f}) pointing (RA/DEC) {1:s}'.format(\
              self.date_start.jyear,Jcurrent_pointing_coords.to_string('hmsdms')))
        
        # derotator
        drot_alpha = a.convert_keyword_coord(self._keywords['HIERARCH ESO INS4 DROT2 RA'][0])
        drot_delta = a.convert_keyword_coord(self._keywords['HIERARCH ESO INS4 DROT2 DEC'][0])
        Jcurrent_drot_coords = coords.SkyCoord(drot_alpha,drot_delta, frame=fk5_timeOfObservation,unit=(u.hourangle, u.deg))
        print('FK5  (ep=J{0:7.2f} eq=J{0:7.2f}) INS4.DROT2.RA/DEC {1:s}'.format(\
              self.date_start.jyear, Jcurrent_drot_coords.to_string('hmsdms')))
        print('Proper motion from the header: pma={0:6.4f}"/yr, pmd={1:6.4f}"/yr'.format(\
              self.pma,self.pmd))

#        location = coords.AltAz(location=EarthLocation.from_geodetic(\
#                                self._longitude, self._latitude, self._altitude),\
#                   obstime=self.date_start,pressure=744*100*u.Pa,\
#                   temperature=10*u.deg_C,relative_humidity=0.05,obswl=6.e-9*u.meter)
#        Jcurrent_pointing_coords_altaz = Jcurrent_pointing_coords.transform_to(location)  

        print(' ')
        if coordOrigin == 'simbad':
            print('Using the (ep=J2000 eq=J{0:7.2f}) Simbad coordinates precessed to J{0:7.2f}'.format(\
                  self.date_start.jyear))
            self.current_coords_target = J2000_current_epoch_simbad_coords.transform_to(fk5_timeOfObservation)
        elif coordOrigin == 'target':
            print('Using the (ep=J2000 eq=J{0:7.2f}) TEL.TARG.ALPHA/DELTA precessed to J{0:8.3f}'.format(self.date_start.jyear))
            self.current_coords_target = J2000_current_epoch_target_coords.transform_to(fk5_timeOfObservation)
        elif coordOrigin == 'pointing':
            print('Using the (ep=J2000 eq=J{0:7.2f}) pointing coordinates (RA/DEC, computed from the guide star) precessed to J{0:8.3f}'.format(self.date_start.jyear))
            self.current_coords_target = Jcurrent_pointing_coords
        elif coordOrigin == 'derot':
            print('Using the coordinates of the target from INS4.DROT2.RA/DEC (derotator)')
            self.current_coords_target = Jcurrent_drot_coords      
                 
        diff_ra = (Jcurrent_pointing_coords.ra-Jcurrent_drot_coords.ra).to(u.arcsec)
        diff_dec = (Jcurrent_pointing_coords.dec-Jcurrent_drot_coords.dec).to(u.arcsec)
#        diff_ampl = np.sqrt(diff_dec**2+(diff_ra/np.cos(Jcurrent_drot_coords.dec))**2)
        diff_ampl = np.sqrt(diff_dec**2+diff_ra**2)
        print('Diff between (ep=J{0:7.2f} eq=J{0:7.2f}) RA/DEC and INS4.DROT2 coordinates : {1:3.1f}" (RA) {2:3.1f}" (DEC) {3:3.1f}" (TOTAL)'.format(\
            self.date_start.jyear,diff_ra.value,diff_dec.value,diff_ampl.value))
        print(' ')

    def compute_properties(self,plot=True):
        """
        It computes the hour angle and parallactic angle for each individual dit.
        It then writes a text file with those values.
        """
        tim_insertion_date = Time('2016-07-13T00:00:00.000')
        if self.date_start < tim_insertion_date:
            print('The data set was obtained prior to the TIM board insertion in SPHERE. There might be errors in the true north.')
        new_DIMM_date = Time('2016-04-02T00:00:00.000')
        if self.date_start < new_DIMM_date:
            print('The data set was obtained prior to the new DIMM. Values of tau0 are therefore to take with caution')
        parang_start_error = np.ndarray(self.getNumberFiles())
        parang_end_error = np.ndarray(self.getNumberFiles())
        derotator_error = np.ndarray(self.getNumberFiles())
        desynchronization_error = np.ndarray(self.getNumberFiles())
        derotator_speed = np.ndarray(self.getNumberFiles())
        hour_angle_array = np.ndarray(self.getNumberFiles())
        alt = np.ndarray(self.getNumberFiles())
        for i,fileName in enumerate(self.getFileNames()):
            fileName = os.path.basename(fileName)
            nb_dit = self._keywords['HIERARCH ESO DET NDIT'][i]   
            t_start = Time(self._keywords['DATE-OBS'][i],location=(self._longitude, self._latitude, self._altitude))
            t_end = Time(self._keywords['DATE'][i],location=(self._longitude, self._latitude, self._altitude))
            t_array = t_start+(t_end-t_start)*(np.arange(0,1.,1./nb_dit)+1./nb_dit/2.)
            lst = t_array.sidereal_time('mean')
            hour_angle = lst - self.current_coords_target.ra
            hour_angle[hour_angle<-12*u.hourangle] += 24*u.hourangle
            hour_angle[hour_angle>12*u.hourangle] -= 24*u.hourangle
            hour_angle_array[i] = np.mean(hour_angle.hour)
            parangle = a.parangle_from_time(t_array,self.current_coords_target)
            ascii.write([t_array.isot,t_array.mjd,t_array.sidereal_time('mean').value,hour_angle.value,parangle.value],
                    os.path.join(self._pathReduc,fileName.replace('.fits','.parang')),names=['date','mjd','lst','hour_angle','par_angle'])
            parang_start_error[i] = np.mod(self._keywords['HIERARCH ESO TEL PARANG START'][i],360)-parangle[0].value
            parang_end_error[i] = np.mod(self._keywords['HIERARCH ESO TEL PARANG END'][i],360)-parangle[-1].value
#            print('Difference between telescope and theoretical parang for file {0:03d}: {1:5.2f} (start) / {2:5.2f} (end)'.format(i,\
#                self._keywords['HIERARCH ESO TEL PARANG START'][i]-parangle[0].value,\
#                self._keywords['HIERARCH ESO TEL PARANG END'][i]-parangle[-1].value))
            alt[i] = self._keywords['HIERARCH ESO TEL ALT'][i]
            parang_start = self._keywords['HIERARCH ESO TEL PARANG START'][i]
            drot_posang = self._keywords['HIERARCH ESO INS4 DROT2 BEGIN'][i]
#            derotator_error[i] = parang_used-parang_real
            #TEL.ALT -2*INS4.DROT2.BEGIN)*pi/180))
            derotator_error[i] =  np.mod(alt[i]-2*drot_posang,360) #- parang_start
            if derotator_error[i]>180:
                derotator_error[i] -= 360 
#            parang_used = np.mod( 2*drot_posang -(alt[i]-parang_start), 180 )
#            parang_real = np.mod(parangle[0].value,180)
#            derotator_error[i] = parang_used-parang_real
            # derotation speed in degree/s.
            time_diff = t_end-t_start
            time_diff.format = u'sec' 
            derotator_speed[i] = (np.mod(self._keywords['HIERARCH ESO INS4 DROT2 BEGIN'][i],360) - \
                              np.mod(self._keywords['HIERARCH ESO INS4 DROT2 END'][i],360)) / time_diff.value               
            desynchronization_error[i] = derotator_error[i]/derotator_speed[i]
            #print('Difference between theoretical and current derotator position angle for file {0:03d}: {1:5.2f}'.format(i,derotator_error[i]))
        mean_derotator_error = np.mean(derotator_error)
        max_derotator_error = np.max(np.abs(derotator_error))
        print('Error between theoretical and current derotator position angle: {0:5.2f} (mean), {1:5.2f} (max)'.format(\
              mean_derotator_error,max_derotator_error))
        if plot:            
            plt.figure(0)
            plt.plot(hour_angle_array,derotator_error)
            plt.xlabel('Hour angle in hour')        
            plt.ylabel('Error $\epsilon$ in degrees')
            plt.savefig(os.path.join(self._pathReduc,self._name+'_derotator_error.pdf'))
            
            plt.figure(1)
            plt.plot(hour_angle_array,desynchronization_error)
            plt.xlabel('Hour angle in hour')        
            plt.ylabel('Desynchronization error in s')
            plt.savefig(os.path.join(self._pathReduc,self._name+'_derotator_desynchronization.pdf'))
            
            plt.figure(2)
            plt.plot(alt,derotator_error)
            plt.xlabel('Altitude in degrees')        
            plt.ylabel('Error $\epsilon$ in degrees')
            plt.savefig(os.path.join(self._pathReduc,self._name+'_derotator_error_vs_altitude.pdf'))

            plt.figure(3)
            plt.plot(hour_angle_array,derotator_speed)
            plt.xlabel('Hour angle in hour')        
            plt.ylabel('Derotator speed in degrees/s')
            plt.close(3)
        mean_parang_start_error = np.mean(parang_start_error)
        max_parang_start_error = np.max(np.abs(parang_start_error))
        mean_parang_end_error = np.mean(parang_end_error)
        max_parang_end_error = np.max(np.abs(parang_end_error))
        print('Difference between telescope and theoretical par. angle at start: {0:5.2f} (mean), {1:5.2f} (max)'.format(mean_parang_start_error,max_parang_start_error))
        print('Difference between telescope and theoretical par. angle at end: {0:5.2f} (mean), {1:5.2f} (max)'.format(mean_parang_end_error,max_parang_end_error))

    def set_wind(self,frameType='O',alt_wind_dir=90.):
        """
        Sets the wind speed and direction at the ground from the keywords of the object.
        Assumes an altitude wind direction east/west unless otherwise specified.
        """
        idFrames = self._get_id_frames(frameType)
        wind_dir = np.asarray([self._keywords['HIERARCH ESO TEL AMBI WINDDIR'][j] for j in idFrames])
        wind_speed = np.asarray([self._keywords['HIERARCH ESO TEL AMBI WINDSP'][j] for j in idFrames])
        wind_dir = np.rad2deg(np.unwrap(np.deg2rad(wind_dir)))
#        wind_dir[wind_dir<180] += 360            
        self.mean_wind_dir = np.mod(np.mean(wind_dir),360)
        self.std_wind_dir = np.std(wind_dir)

        self.mean_wind_speed = np.mean(wind_speed)
        self.std_wind_speed = np.std(wind_speed)        

        self.alt_wind_dir = alt_wind_dir

        print('Mean wind speed (ground): {0:.1f} +/- {1:.1f} m/s'.format(self.mean_wind_speed,self.std_wind_speed))
        print('Mean wind direction     : {0:.1f} +/- {1:.1f} degrees'.format(self.mean_wind_dir,self.std_wind_dir))
        print('Altitude wind direction : {0:.1f} degrees'.format(self.alt_wind_dir))
        return 

    def get_parang(self,frameType='all',save=False):
        """
        Retrieve the parallactic angles of all files and optionnally save them to
        a fits file along with the mjd and hour angle.
        Input:
            -frameType: 'all' if all files from the file list are to be processed,
                or 'O' (OBJECT) (resp. 'F' (FLUX), 'C' (CENTER)) if only the
                object (resp. center, flux) files are to be processed.
            - save: if true saves a fits file with the par angle, mjd and hour angle
        Output
            - tuple (parang,hour_angle,mjd) 
        """
        idFrames = self._get_id_frames(frameType)
#        totalFrames = len(idFrames)
        totalFrames = np.sum([self._keywords['HIERARCH ESO DET NDIT'][j] for j in idFrames])
        parang = np.ndarray((totalFrames))
        mjd = np.ndarray((totalFrames))
        hour_angle = np.ndarray((totalFrames))
        counter = 0
        for i,fileName in enumerate([os.path.basename(self._fileNames[j]) for j in idFrames]):
            ndit = self._keywords['HIERARCH ESO DET NDIT'][idFrames[i]]
            data_angles = ascii.read(os.path.join(self._pathReduc,fileName.replace('.fits','.parang')))
            parang[counter:counter+ndit] = data_angles['par_angle']
            hour_angle[counter:counter+ndit] = data_angles['hour_angle']
            mjd[counter:counter+ndit] = data_angles['mjd']
            counter = counter + ndit        
        parang=np.rad2deg(np.unwrap(np.deg2rad(parang)))
        if frameType == 'O':
            print('Parallactic angle variation of {0:.1f} from {1:.1f} to {2:.1f}'.format(parang[-1]-parang[0],parang[0],parang[-1])) 
        if save:
            fits.writeto(os.path.join(self._pathReduc,self._name+'_parang_'+frameType+'.fits'),parang,clobber=True,output_verify='ignore')
            fits.writeto(os.path.join(self._pathReduc,self._name+'_hourangle_'+frameType+'.fits'),hour_angle,clobber=True,output_verify='ignore')
            fits.writeto(os.path.join(self._pathReduc,self._name+'_mjd_'+frameType+'.fits'),mjd,clobber=True,output_verify='ignore')
        return parang,hour_angle,mjd
        
    def _get_id_frames(self,frameType='all'):
        """
        Internal function that returns a list with the indices of the requested frames
        (either "all", "O", "C" or "F")
        """
        if frameType == 'all':
            idFrames = range(self.getNumberFiles())
        else:
            dpr_type = self._keywords['HIERARCH ESO DPR TYPE']
            idFrames = []
            if frameType == 'O':
                for index,dpr in enumerate(dpr_type):
                    if dpr == "OBJECT":
                        idFrames.append(index)
            elif frameType == 'F':
                for index,dpr in enumerate(dpr_type):
                    if dpr == "OBJECT,FLUX":
                        idFrames.append(index)
            elif frameType == 'C':
                for index,dpr in enumerate(dpr_type):
                    if dpr == "OBJECT,CENTER":
                        idFrames.append(index)
            else:
                raise TypeError('The frameType keyword must be "all", "O" (for "OBJECT"), "F" (for "OBJECT,FLUX") or "C" (for "OBJECT,CENTER")')
        return idFrames

    def get_psf_frames(self,size=None,camera='left',fwhm=None):
        """
        Builds the cube of PSF frames ('F'), then recenters each frame using 
        a 2D gaussian fit.
        """
        if size == None:
            size = self._rowNb
        if fwhm==None:
            if camera=='left':
                fwhm = self.theoretical_fwhm[0]
            elif camera=='right':
                fwhm = self.theoretical_fwhm[1]
        dist_center = distance_array([size,size],verbose=False)
        cube_psf,_,_ = self.write_master_cube(camera=camera,centerxy=[self._rowNb/2,self._columnNb/2],\
                     size=size,frameType='F',output=True)
        if len(cube_psf.shape)==3:     
            median_psf = np.median(cube_psf,axis=0)
#            nframes_psf = cube_psf.shape[0]
        else:
            median_psf = cube_psf
#            nframes_psf = 1            
        posmax = np.argmax(median_psf * (dist_center<self._rowNb/4))
        posy,posx = np.unravel_index(posmax,median_psf.shape)
#        vip/preproc/recentering.cube_recenter_gauss2d_fit(array, xy, fwhm=4, subi_size=5, nproc=1,
        recentered_cube = vip.preproc.recentering.cube_recenter_gauss2d_fit(cube_psf,\
            [posx,posy], fwhm=fwhm, subi_size=5,nproc=1,full_output=False,verbose=True,\
            save_shifts=False, offset=None, negative=False, debug=False)
        return recentered_cube
        
    def get_psf_scaling_factor(self,verbose=True):
        """
        Returns a tuple with the scaling factor between the OBJECT,FLUX frames 
        and the OBJECT frames based on the difference in DIT and ND filter.
        Each dimension corresponds to the left and right  camera respectively.
        """
        idFlux = self._get_id_frames('F')
        idObject = self._get_id_frames('O')
        dit_list_flux = [self._keywords['HIERARCH ESO DET SEQ1 DIT'][i] for i in idFlux]
        dit_list_object = [self._keywords['HIERARCH ESO DET SEQ1 DIT'][i] for i in idObject]
        bb_filter_list_flux = [self._keywords['HIERARCH ESO INS1 FILT NAME'][i] for i in idFlux]
        db_filter_list_flux =[self._keywords['HIERARCH ESO INS1 OPTI2 NAME'][i] for i in idFlux]
        nd_filter_list_flux = [self._keywords['HIERARCH ESO INS4 FILT2 ID'][i] for i in idFlux]
        nd_filter_list_object = [self._keywords['HIERARCH ESO INS4 FILT2 ID'][i] for i in idObject]
        for dit in dit_list_flux:
            if dit != dit_list_flux[0]:
                raise ValueError('All HIERARCH ESO DET SEQ1 DIT are not identical for the OBJECT,FLUX frames')       
        for dit in dit_list_object:
            if dit != dit_list_object[0]:
                raise ValueError('All HIERARCH ESO DET SEQ1 DIT are not identical for the OBJECT frames')
        for nd_filter_flux in nd_filter_list_flux:
            if nd_filter_flux != nd_filter_list_flux[0]:
                raise ValueError('All HIERARCH ESO INS4 FILT2 ID are not identical for the OBJECT,FLUX frames')
        for nd_filter_object in nd_filter_list_object:
            if nd_filter_object != nd_filter_list_object[0]:
                raise ValueError('All HIERARCH ESO INS4 FILT2 ID are not identical for the OBJECT frames')
        if db_filter_list_flux[0].startswith('CLEAR'):
            db_filter = None
        else:
            db_filter = db_filter_list_flux[0]
        if nd_filter_list_flux[0].startswith('FILT_ND'):
            nd_flux = float(nd_filter_list_flux[0][8:])
        else:
            nd_flux = 0.
        if nd_filter_list_object[0].startswith('FILT_ND'):
            nd_object = float(nd_filter_list_object[0][8:])
        else:
            nd_object = 0.
        transmission_flux = sph.sphere_transmission(BB_filter=bb_filter_list_flux[0], DB_filter=db_filter, NDset=nd_flux)
        transmission_object = sph.sphere_transmission(BB_filter=bb_filter_list_flux[0], DB_filter=db_filter, NDset=nd_object)
        if verbose:
            print('OBJECT,FLUX: DIT={0:4.2f} ND_{1:3.1f}'.format(dit_list_flux[0],nd_flux))
            print('OBJECT     : DIT={0:4.2f} ND_{1:3.1f}'.format(dit_list_object[0],nd_object))
            print('Scaling factor (left) = {0:6.1f} = {1:6.1f} (DIT) x {2:6.1f} (ND)'.format(\
                dit_list_object[0]/dit_list_flux[0]*transmission_object[0]/transmission_flux[0],\
                dit_list_object[0]/dit_list_flux[0],transmission_object[0]/transmission_flux[0]))
        scaling_factor = dit_list_object[0]/dit_list_flux[0]*transmission_object/transmission_flux
        return scaling_factor

    def get_theoretical_fwhm(self,verbose=True):
        """
        Reads the INS.COMB.FILT and returns the corrresponding theoretical FWHM
        """
        filter_list = self._keywords['HIERARCH ESO INS COMB IFLT']
        for filt in filter_list:
            if filt != filter_list[0]:
                raise ValueError('All HIERARCH ESO INS COMB IFLT are not identical')                
        return sph.theoretical_sphere_fwhm(filter_name=filter_list[0],verbose=True)

    def get_region_file(self,size=1024,save=True):
        """
        Creates a region file to open in DS9 containing arrows with the wind direction, 
        axis of the DM...
        """
        center=size//2+1
        length=size//2*0.95
        parang,hour_angle,mjd = self.get_parang(frameType='O',save=False)
        parang_start = parang[0]
        parang_end = parang[-1]
        parang_mean = np.mean(parang)
        parang_var = np.max(parang)-np.min(parang)
        # The spiders are at PA_detector = 50 / 130 / 230 / 310 (-50) 
        # 40deg from horizontal
        paDetectorSpider = np.array([50,130,230,310])
        paOnSkySpider_mean = np.mod(paDetectorSpider + (parang_mean - self.true_north - self.pupil_offset),360)
        paOnSkySpider_start = np.mod(paDetectorSpider + (parang_start - self.true_north - self.pupil_offset),360)
        paOnSkySpider_end = np.mod(paDetectorSpider + (parang_end - self.true_north - self.pupil_offset),360)
        paDMaxis = np.array([90,270])
        paOnSkyDMaxis_mean = np.mod(paDMaxis + (parang_mean - self.true_north - self.pupil_offset),360)
        reg_string=\
        """# Region file format: DS9 version 4.1
        global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
        physical
        # vector({0:d},{1:d},{2:.1f},{3:4.1f}) vector=1 color=red text={{spiders}}
        # vector({0:d},{1:d},{2:.1f},{4:4.1f}) vector=1 color=red dash=1
        # vector({0:d},{1:d},{2:.1f},{5:4.1f}) vector=1 color=red dash=1
        # vector({0:d},{1:d},{2:.1f},{6:4.1f}) vector=1 color=red
        # vector({0:d},{1:d},{2:.1f},{7:4.1f}) vector=1 color=red
        # vector({0:d},{1:d},{2:.1f},{8:4.1f}) vector=1 color=red
        # text({9:d},{10:.1f}) color=white font="helvetica 16 normal roman" text={{Parang. variations: {11:4.1f} deg}}
        # vector({0:d},{1:d},{2:.1f},{12:4.1f}) vector=1 color=blue text={{DM axis}}
        # vector({0:d},{1:d},{2:.1f},{13:4.1f}) vector=1 color=blue
        # vector({0:d},{1:d},{2:.1f},{14:4.1f}) vector=1 color=white text={{alt. wind}}
        # vector({0:d},{1:d},{2:.1f},{15:4.1f}) vector=1 color=yellow text={{tel. wind}}
        """.format(center,center,length,paOnSkySpider_mean[0]+90,\
            paOnSkySpider_start[0]+90,\
            paOnSkySpider_end[0]+90,\
            paOnSkySpider_mean[1]+90,\
            paOnSkySpider_mean[2]+90,
            paOnSkySpider_mean[3]+90,\
            center,size*0.98,parang_var,\
            paOnSkyDMaxis_mean[0]+90,paOnSkyDMaxis_mean[1]+90,\
            self.alt_wind_dir+90,self.mean_wind_dir+90)
        if save:
            filename = os.path.join(self._pathReduc,\
            '{0:s}_arrows_{1:03d}x{1:03d}.reg'.format(self._name,size))
            txtfile = open(filename,'w')
            txtfile.write(reg_string)
            txtfile.close()   
        if self.alt_wind_dir==90:
            print('The altitude wind is probably the 90deg guess value. You can change that by calling the function analyse_sparta(pathSpartaNight,debug=False,force=True) and then compute_statistics(self,frameType="O")')
        return reg_string
               
    def write_master_cube(self,camera='left',centerxy=None,size=None,frameType='all',
                          output=False,dithering=True):
        """
        Reads the fits files processed by the pipeline and recenters it optionally.
        Create a master cube.
        Input:
            - camera: 'left' or 'right'
            - centerxy: a 2-element list with the coordinate [x,y] of the center 
                        of the image in the pipeline processed file
            - size: the desired output size of the cube. If None, by default the
                    cube keeps the original dimension (512x512)
            -frameType: 'all' if all files from the file list are to be processed,
                        or 'O' (OBJECT) (resp. 'F' (FLUX), 'C' (CENTER)) if only the
                        object (resp. center, flux) files are to be processed.
            -output: boolean (False by default) to return the cube and parallactic angles and 
                    and derotation angles (parallactic angles - pupil offset - true north)
        """
        if camera != 'left' and camera != 'right':
            raise TypeError('The camera keyword must be "left" or "right". Got {0}'.format(camera))
        if centerxy is None:
#            centerxy = [self._columnNb/2,self._rowNb/2]
            if camera == 'left':
                centerxy = self._centerxy[0,:]
            elif camera == 'right':
                centerxy = self._centerxy[1,:]
#            print('No center provided. Will center about {0:4d} {1:4d}'.format(centerxy[0],centerxy[1]))
        if size == None:
            size = self._rowNb
        if len(centerxy) != 2:
            raise TypeError('The center must be a list of 2 integers. Got {0}'.format(centerxy))
        idFrames = self._get_id_frames(frameType)
        totalFrames = self.getTotalNumberFrames(frameType=frameType)
        print('Creating the master cube of {0:3d} frames of size {1:4d}x{2:4d}...'.format(totalFrames,size,size))
        cube = np.ndarray([totalFrames,size,size])
        cube.fill(np.nan)
        parang = np.ndarray((totalFrames))
        mjd = np.ndarray((totalFrames))
        hour_angle = np.ndarray((totalFrames))
        counter = 0
        print('Cropping and recentering the frames...')
        for i,fileName in enumerate([os.path.basename(self._fileNames[j]) for j in idFrames]):
            print('Reading {0:s}'.format(fileName))
            ndit = self._keywords['HIERARCH ESO DET NDIT'][idFrames[i]]
            pipelineName = os.path.join(self._pathReduc,fileName.replace('.fits','_'+camera+'.fits'))
            data_angles = ascii.read(os.path.join(self._pathReduc,fileName.replace('.fits','.parang')))
            parang[counter:counter+ndit] = data_angles['par_angle']
            hour_angle[counter:counter+ndit] = data_angles['hour_angle']
            mjd[counter:counter+ndit] = data_angles['mjd']
            cube_pipeline = fits.getdata(pipelineName)
            if dithering:
                centerx = int(centerxy[0]+self._keywords['HIERARCH ESO INS1 DITH POSX'][idFrames[i]])
                centery = int(centerxy[1]+self._keywords['HIERARCH ESO INS1 DITH POSY'][idFrames[i]])            
            else:
                centerx = int(centerxy[0])
                centery = int(centerxy[1])                            
            original_ll_x = np.max([0,centerx-size/2])
            original_ll_y = np.max([0,centery-size/2])
            if np.mod(size,2) == 0:
                original_ur_x = np.min([self._columnNb,centerx+size/2])
                original_ur_y = np.min([self._rowNb,centery+size/2])
            else:
                original_ur_x = np.min([self._columnNb,centerx+size/2+1])
                original_ur_y = np.min([self._rowNb,centery+size/2+1])                
            targetSpan_x = original_ur_x - original_ll_x 
            targetSpan_y = original_ur_y - original_ll_y 
            if original_ll_x > 0:
                target_ll_x = 0
            else:
                target_ll_x = size/2-centerx
            if original_ll_y > 0:
                target_ll_y = 0
            else:
                target_ll_y = size/2 - centery
            cube[counter:counter+ndit,target_ll_y:target_ll_y+targetSpan_y,target_ll_x:target_ll_x+targetSpan_x] = \
                cube_pipeline[:,original_ll_y:original_ur_y,original_ll_x:original_ur_x]
            subpixel_shift = np.fix(centerxy) - centerxy  #remaining shift
            if subpixel_shift[0] != 0. or subpixel_shift[1] != 0:
                for k in range(ndit):
                    tmp = cube[counter+k,:,:]
                    cube[counter+k,:,:] = vip.preproc.recentering.frame_shift(tmp, subpixel_shift[1], subpixel_shift[0],\
                        imlib='ndimage-fourier')#, interpolation='bicubic')
            counter = counter + ndit
        derotation_angles = parang-self.true_north-self.pupil_offset
        fits.writeto(os.path.join(self._pathReduc,self._name+'_{0:03d}x{1:03d}_'.format(size,size)+camera+'_'+frameType+'.fits'),cube,header=self.firstHeader,clobber=True,output_verify='ignore')
        fits.writeto(os.path.join(self._pathReduc,self._name+'_parang_'+frameType+'.fits'),parang,clobber=True,output_verify='ignore')
        fits.writeto(os.path.join(self._pathReduc,self._name+'_derotation_angles_'+frameType+'.fits'),derotation_angles,clobber=True,output_verify='ignore')
        fits.writeto(os.path.join(self._pathReduc,self._name+'_ha_'+frameType+'.fits'),hour_angle,clobber=True,output_verify='ignore')
        fits.writeto(os.path.join(self._pathReduc,self._name+'_mjd_'+frameType+'.fits'),mjd,clobber=True,output_verify='ignore')
        print('First frame: parang {0:6.1f},hour angle{1:6.1f}'.format(parang[0],hour_angle[0]))
        print('Last  frame: parang {0:6.1f},hour angle{1:6.1f}'.format(parang[-1],hour_angle[-1]))
        print('Delta      : parang {0:6.1f},hour angle{1:6.1f}'.format(np.abs(parang[0]-parang[-1]),np.abs(hour_angle[0]-hour_angle[-1])))
        if output:
            return cube,parang,derotation_angles
            
    def get_true_north(self,plot=True,interpolate=False):
        data_tn = ascii.read(os.path.join(path_data,'true_north.txt'))
        array_mjd = Time(data_tn['date']).mjd
        f = interp1d(array_mjd,data_tn['true_north'])
        mjd_start = np.min(array_mjd)
        mjd_end = np.max(array_mjd)
        x = np.linspace(mjd_start, mjd_end, num=101, endpoint=True)
        mean_true_north = np.median(data_tn['true_north']) #-1.75 # Mean true north value
        try:
            interpolated_true_north = float(f(self.date_start.mjd))
            good_interpolation = True
        except ValueError as e:
            good_interpolation = False            
            if self.date_start.mjd < mjd_start:
                arg_min = np.argmin(array_mjd)
                print('There is no true north measurement before the date of observation. We used the same TN as the first measurement: {0:5.2f}degrees measured on {1:s}'.format(\
                    data_tn['true_north'][arg_min],data_tn['date'][arg_min]))
            elif self.date_start.mjd > mjd_end:
                arg_max = np.argmax(array_mjd)
                print('There is no true north measurement after the date of observation. We used the same TN as the last measurement: {0:5.2f}degrees measured on {1:s}'.format(\
                    data_tn['true_north'][arg_max],data_tn['date'][arg_max]))
            else:
                raise e
        if plot:
            plt.figure(5)
            plt.plot(array_mjd,data_tn['true_north'], 'bo',label='astrometric meas.')
            plt.plot(x, f(x), 'b-')
            plt.plot([mjd_start,mjd_end],[mean_true_north,mean_true_north],'r-',label='mean value')
            if good_interpolation:
                plt.plot([self.date_start.mjd],[interpolated_true_north],'ro',label='interpolated value')
            plt.legend(frameon=False,loc='best')
            plt.xlabel('MJD')
            plt.ylabel('True north in degrees')
        if interpolate:
            self.true_north=interpolated_true_north
            print('You chose the interpolated true north of {0:6.2f} deg'.format(self.true_north))
        else:
            self.true_north=mean_true_north            
            print('You chose the mean true north of {0:6.2f} deg'.format(self.true_north))
        return self.true_north
        
    def analyse_sparta(self,folder,debug=False,force=False):
        """
        Function that reads the sparta files (DPR.TYPE=OBJECT,AO) contained in the input folder
        and analyses the performance of the system and the atmospheric conditions. 
        It plots a summary of the atmospheric conditions in a pdf file and saves many 
        csv files. 
        Input:
            - folder: folder where the sparta files are located. This folder can contain additional
                sparta files from the same day (and it is recommended to get an overview of
                the night)
            - debug: to print additional information of the night
            - force: if False, check if the plot_sparta_data function was already run and in this
                case do not repeat it.  
        """
        if force or len(glob.glob(os.path.join(folder,'summary*.csv')))==0: 
            if debug==False:
                print('Script on-going... be patient')
            plot_sparta_data.plot_sparta_data(path_raw=folder,path_output=folder,plot=True,debug=debug)
        elif len(glob.glob(os.path.join(folder,'summary*.csv')))==1: 
            print('The folder {0:s} already contains the summary file {1:s}, so the script to extract data from sparta was not run'.format(folder,glob.glob(os.path.join(folder,'summary*.csv'))[0]))
            print('If you want to run it anyway and overwrite the current output, use the option force=True')
        else:
            print('The folder {0:s} contains {1:d} summary*.csv files. This is not expected.'.format(folder,len(glob.glob(os.path.join(folder,'summary*.csv')))))

        t_start = Time(self._keywords['DATE-OBS'])
        t_end = Time(self._keywords['DATE'])
        t_mean = t_start+(t_end-t_start)/2.
        for file_to_read in self._files_to_read:
            # we read the file if it exists and is unique:
            if len(glob.glob(os.path.join(folder,file_to_read)))==1:
                param_file = glob.glob(os.path.join(folder,file_to_read))[0]
                pd_params = pd.read_csv(param_file)
                if len(pd_params.keys())<2:
                    print('No data to be read in {0:s}'.format(param_file))
                else:
                    print('Reading {0:s}'.format(file_to_read))
                    time_params = Time(list(pd_params['date']))#,format='isot')
                    for key in pd_params.keys():
                        isNumber_array = [isinstance(val,(int, float)) for val in np.unique(pd_params[key])]
                        if (key not in ['date'] and np.all(isNumber_array)):
#                            self._keywordList.append(key)
                            self._keywordsExtra[key]=interpolate_date(time_params,pd_params[key],t_mean,plot=False,kind='linear')
            elif len(glob.glob(os.path.join(folder,file_to_read)))==0:
                print('No file {0:s}'.format(file_to_read))
            else:
                print('The folder {0:s} contains {1:d} {2:s} files. This is not expected.'.format(\
                      folder,len(glob.glob(os.path.join(folder,file_to_read))),
                      file_to_read))
        return

    def compute_statistics(self,frameType='all'):
        """
        Compute the mean, stdev, min and max values of the different numerical
        keywords values and sparta values. It saves the output in a csv file
        called filename_keywords_statistics_*.csv
        Input:
            - frameType: all for all frames, O for only the object frames, or 
            F for the Flux frames or C for the center frames.
        """
        idFrames = self._get_id_frames(frameType)      
        if len(idFrames)==0:
            print('There is no {0:s} frames !! Returning'.format(frameType))
            return
        name=[]
        mean = []
        rms = []
        med = []
        max_val = []
        min_val = []
        for key in self._keywordList:
            if isinstance(self._keywords[key][0],(int, float)): 
                name.append(key)
                array = np.asarray([self._keywords[key][i] for i in idFrames])
                mean.append(np.nanmean(array))
                rms.append(np.nanstd(array))
                med.append(np.nanmedian(array))
                max_val.append(np.nanmax(array))
                min_val.append(np.nanmin(array))
        for key in self._keywordsExtra.keys():
            if isinstance(self._keywordsExtra[key][0],(int, float)): 
                name.append(key)
                array = np.asarray([self._keywordsExtra[key][i] for i in idFrames])
                mean.append(np.nanmean(array))
                rms.append(np.nanstd(array))
                med.append(np.nanmedian(array))
                max_val.append(np.nanmax(array))
                min_val.append(np.nanmin(array))   
            if 'ecmwf_200mbar_winddir' in key and frameType=='O':
                self.alt_wind_dir = np.nanmean(array)
                print('The altitude wind direction was set to {0:.1f} deg.'.format(self.alt_wind_dir))
        filename = os.path.join(self._pathReduc,\
            '{0:s}_keywords_statistics_{1:s}.csv'.format(self._name,frameType))
        ascii.write([name,mean,med,rms,max_val,min_val],\
            filename,names=['name','mean','median','rms','max','min'],\
            format='csv',overwrite=True)
        print('Wrote the statistics file {0:s}'.format(filename))
        return

    def interpolate_metadata(self,frameType='O',spartafolder=None):
        """
        Interpolates all possible metadata to the level of the individual frames.
        It writes the result in a csv file called 
        Input:
            -frameType: 'all' if all files from the file list are to be processed,
                        or 'O' (OBJECT) (resp. 'F' (FLUX), 'C' (CENTER)) if only the
                        object (resp. center, flux) files are to be processed.
            -spartafolder: folder where the sparta csv files are located.
        Output:
        """
        idFrames = self._get_id_frames(frameType)
        totalFrames = self.getTotalNumberFrames(frameType=frameType)
        ndit_array = [self._keywords['HIERARCH ESO DET NDIT'][idx] for idx in  idFrames]
        # basic data read from the *.parang file
        parang = np.ndarray((totalFrames))
        mjd = np.ndarray((totalFrames))
        hour_angle = np.ndarray((totalFrames))
        counter = 0
        for i,fileName in enumerate([os.path.basename(self._fileNames[j]) for j in idFrames]):
            ndit = self._keywords['HIERARCH ESO DET NDIT'][idFrames[i]]
            data_angles = ascii.read(os.path.join(self._pathReduc,fileName.replace('.fits','.parang')))
            parang[counter:counter+ndit] = data_angles['par_angle']
            hour_angle[counter:counter+ndit] = data_angles['hour_angle']
            mjd[counter:counter+ndit] = data_angles['mjd']
            counter = counter + ndit   
        # We now create a panda array that will contain all data
        time_frames = Time(mjd,format='mjd',scale='utc')
        time_frames.format='isot'
        pd_metadata_frames = pd.DataFrame({'date':time_frames,'par_angle':parang,\
                                           'hour_angle':hour_angle,'mjd':mjd,\
                                           })
        # We now add all informations from the different keywords
        for key in self._keywordList:
            array_per_file = np.asarray([self._keywords[key][i] for i in idFrames])
            
            array_per_frame = [x for idx,item in enumerate(array_per_file) for x in repeat(item, ndit_array[idx])]
            pd_metadata_frames['HEADER '+key]=array_per_frame

        # We now add the informations from sparta in case there are available
        if spartafolder is not None:
            for file_to_read in self._files_to_read:
                # we read the file if it exists and is unique:
                if len(glob.glob(os.path.join(spartafolder,file_to_read)))==1:
                    param_file = glob.glob(os.path.join(spartafolder,file_to_read))[0]
                    pd_params = pd.read_csv(param_file)
                    if len(pd_params.keys())>1:
                        time_params = Time(list(pd_params['date']))#,format='isot')
                        for key in pd_params.keys():
                            isNumber_array = [isinstance(val,(int, float)) for val in np.unique(pd_params[key])]
                            # if it is a numerical parameter, we interpolate it linearly
                            if key != 'date' and np.all(isNumber_array):
                                array_per_frame = interpolate_date(time_params,pd_params[key],time_frames,plot=False,kind='linear')    
                                pd_metadata_frames[key]=array_per_frame
        filename = os.path.join(self._pathReduc,\
            '{0:s}_metadata_{1:s}.csv'.format(self._name,frameType))
        pd_metadata_frames.to_csv(filename)
#        pdb.set_trace()
        print('Wrote the metadata file {0:s}'.format(filename))


if __name__=='__main__':  
    target_name = 'HD105'
    pathTarget = os.path.join('/Volumes/SHARDDS_data/survey_disk',target_name)
    pathRaw = os.path.join(pathTarget,'raw')
    pathOut = os.path.join(pathTarget,'pipeline')        
    fileNames = 'SPHER.*.fits'
    irdis_data = IrdisDataHandler(pathRaw,pathOut,fileNames,name=target_name,coordOrigin='derot')            
    