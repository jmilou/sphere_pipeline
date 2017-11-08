# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:49:10 2016

@author: jmilli
"""
import numpy as np
import astropy.units as u
latitude_UT3 = -(24.+37./60.+30.300/3600.)*u.degree
longitude_UT3 = -(70.+24./60.+9.896/3600)*u.degree

def parangle(ha,dec,lat=-24.6253):
    """
    Computes the parallactic angle
    Input:
        - ha: hour angle in decimal hour
        - dec: declination in degrees 
        - lat: latitude of observations in degrees (default is Paranal)
    """
    ha_rad = np.deg2rad(ha*15.)
    dec_rad = np.deg2rad(dec)
    lat_rad = np.deg2rad(lat)
    p = np.arctan( -np.sin(ha_rad) / (np.sin(dec_rad)*np.cos(ha_rad) - np.cos(dec_rad)*np.tan(lat_rad)))
    denom = np.sin(dec_rad)*np.cos(ha_rad) - np.cos(dec_rad)*np.tan(lat_rad)
    return np.rad2deg(p + (denom>0)*np.pi)

#def parangle_from_time(time,coords,latitude=-(24.+37./60.+30.300/3600.)*u.degree,longitude=-(70.+24./60.+9.896/3600)*u.degree)
def parangle_from_time(time,coords):
    """
    Computes the parallactic angle
    Input:
        - time: an instance of the Time class (astropy.time.Time) with a correct
            location for the sideral time computation
        - coords: an instance of the coordinates class (astropy.coordinates.SkyCoord)
    Returns the parallactic angle in degrees
    """
    sideral_time = time.sidereal_time('mean')#time.sidereal_time('apparent')
    hour_angle = sideral_time - coords.ra
    y = np.sin(hour_angle)
    x = np.tan(time.location.latitude) * np.cos(coords.dec ) - np.sin(coords.dec) * np.cos(hour_angle)
    parangle = np.rad2deg(np.arctan2( y, x ))
    return np.mod(parangle,360*u.degree)


def zenangle(ha,dec,lat=-24.6253):
    """
    Determine Zenith angle
    Input:
        - ha: hour angle in decimal hour
        - dec: declination in degrees 
        - lat: latitude of observations in degrees (default is Paranal)
    """
    ha_rad = np.deg2rad(ha*15.)
    dec_rad = np.deg2rad(dec)
    lat_rad = np.deg2rad(lat)
    za = np.arccos ( np.sin(lat_rad)*np.sin(dec_rad)+np.cos(lat_rad)*np.cos(dec_rad)*np.cos(ha_rad))
    return np.rad2deg(za)
    
def azangle(ha,dec,lat=-24.6253):
    """
    Determine Azimuth
    Input:
        - ha: hour angle in decimal hour
        - dec: declination in degrees 
        - lat: latitude of observations in degrees (default is Paranal)
    """
    ha_rad = np.deg2rad(ha*15.)
    dec_rad = np.deg2rad(dec)
    lat_rad = np.deg2rad(lat)
    za = np.arccos ( np.sin(lat_rad)*np.sin(dec_rad)+np.cos(lat_rad)*np.cos(dec_rad)*np.cos(ha_rad))
    az = np.arccos((np.sin(dec_rad)-np.sin(lat_rad)*np.cos(za))/(np.cos(lat_rad)*np.sin(za)))
    return np.rad2deg(az)
    
def convert_keyword_coord(keyword_coord):
    """
    Convert a keyword of type -124700.06 into a string "-12:47:00.06" readable
    by astropy.coordinates
    Input:
        - keyword_coord of type float (example -124700.06 as retrived from
        header['HIERARCH ESO TEL TARG ALPHA'] or header['HIERARCH ESO TEL TARG DEC'])
    Output:
        - formatted string directly readable by astropy.coordinates (example: "-12:47:00.06")
    """
    if type(keyword_coord) != float:
        raise TypeError('The argument {0} is not a float'.format(keyword_coord))
    if keyword_coord<0:
        keyword_coord_str = '{0:012.4f}'.format(keyword_coord) #0 padding with 6 digits (11-4-1-1) for the part before the point.
        keyword_formatted = '{0:s}:{1:s}:{2:s} '.format(keyword_coord_str[0:3],
                keyword_coord_str[3:5],keyword_coord_str[5:])
    else:
        keyword_coord_str = '{0:011.4f}'.format(keyword_coord) #0 padding with 6 digits (11-4-1) for the part before the point.
        keyword_formatted = '{0:s}:{1:s}:{2:s} '.format(keyword_coord_str[0:2],
                keyword_coord_str[2:4],keyword_coord_str[4:])
    return keyword_formatted