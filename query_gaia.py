#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 08:47:58 2023

@author: millij
"""


from astropy.io import fits
import argparse
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import ICRS
import pandas as pd
from pathlib import Path
import find_target_name_from_header as f
     
def get_gaia_dr3_id(simbad_dico,verbose=True):
    """
    Returns the GAIA DR3 ID of a target
    Input: 
        - simbad_dico: a dictionnary formatted like the ouput of 
          find_target_name_from_header.query_simbad
        - verbose: bool to get verbose (optionnal, True by default)
    """
    simbad_names = str(simbad_dico['simbad_IDS'])
    start_index = simbad_names.index('Gaia DR3')+8
    last_index = simbad_names[start_index:].index('|')+start_index
    id_gaia_dr3 = int(simbad_names[start_index:last_index].strip())
    if verbose:
        print('The target GAIA DR3 ID is {0:d}'.format(id_gaia_dr3))
    return id_gaia_dr3

def propagate_pm_to_obs_date(id_gaia_dr3,date_obs,verbose=True):
    """
    This function propagates the coordinate of the target identified by its GAIA DR3 ID 
    to the epoch of observation.
    Input: 
        - id_gaia_dr3: int 
        - date_obs: astropy.time.Time object
        - verbose: bool to get verbose (optionnal, True by default)
    """
    query1 = """SELECT 
            EPOCH_PROP(gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.pmra,gaia_source.pmdec,gaia_source.radial_velocity,2015.5,{0:.6f}) 
            FROM gaiadr3.gaia_source 
            WHERE source_id = '{1:d}'""".format(date_obs.jyear,id_gaia_dr3)
    
    j1 = Gaia.launch_job(query1)
    gaia_prop_result = j1.get_results()
    return gaia_prop_result

def query_gaia_sources(gaia_prop_result,search_radius_arcsec=6.25,verbose=True):
    """
    This function queries all GAIA DR3 sources within a given radius from the source 
    at the epoch of observations (assuming the sources do not have signifcant 
    proper motion) 
    Input:
        - gaia_prop_result
        - search_radius_arcsec: float. The search radius in arcsec, 6.25" for IRDIS by default
          should be changed to 0.8 for the IFS
        - verbose: bool to get verbose (optionnal, True by default)
    """
    search_radius_degree = search_radius_arcsec/3600.
    query2 = """SELECT DISTANCE(
      POINT('ICRS', ra, dec),
      POINT('ICRS', {0:f}, {1:f})) AS dist, *
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
      POINT('ICRS', ra, dec),
      CIRCLE('ICRS', {0:f}, {1:f}, {2:f}))
    ORDER BY dist ASC""".format(gaia_prop_result['EPOCH_PROP'][0][0],\
        gaia_prop_result['EPOCH_PROP'][0][1],search_radius_degree)
    
    j2 = Gaia.launch_job(query2)
    gaia_sources_result = j2.get_results()
    nstars2 = len(gaia_sources_result)
    if verbose:
        print('\nThe GAIA request found {0:d} sources within {1:.2f} arcsec \n'.format(nstars2,search_radius_arcsec))
        # print(gaia_sources_result)
        # print(gaia_sources_result.keys())        
    return gaia_sources_result

def propagat_pm_to_obs_date(gaia_sources_result,gaia_prop_result,id_gaia_dr3,date_obs,verbose=True):
    """
    For each source found, we propagate the proper motion from 2015.5 to the 
    epoch of observation and derive the distance between the target and the contaminant.     
    Input:
        - gaia_sources_result: a dictionnary (the output of the function query_gaia_sources)
        - gaia_prop_result: a dictionnary (the output of the function propagate_pm_to_obs_date) 
        - id_gaia_dr3: int (the output of the function get_gaia_dr3_id)
        - date_obs: astropy.time.Time object
        - verbose: bool to get verbose (optionnal, True by default)
        
    Returns
    -------
    A dictionnary with the contaminants from Gaia DR3 

    """
    dico_contaminants = {\
        'ID Gaia DR3':[],'RA (deg)':[],'DEC (deg)':[],'pmra':[],'pmdec':[],\
        'delta RA (arcsec)':[],'delta DEC (arcsec)':[],\
        'separation (arcsec)':[],'PA (deg)':[],'parallax':[],\
        'phot_g_mean_mag':[],'phot_bp_mean_mag':[],'phot_rp_mean_mag':[],\
        'FLUX_J':[],'FLUX_H':[],'FLUX_K':[]\
        }

    fields = ['flux(V)','flux(G)','flux(R)','flux(J)','flux(H)','flux(K)',\
              'flux_error(V)','flux_error(G)','flux_error(R)','flux_error(J)',\
              'flux_error(H)','flux_error(K)',\
              'sp','pmra','pmdec',\
              'ra(gal)','dec(gal)',\
              'pm_err_angle','pm_err_maja','pm_err_mina','pm_bibcode','pm_qual',\
              'plx','plx_error','plx_bibcode',\
              'rv_value','rvz_error','rvz_qual','rvz_bibcode','rvz_radvel']
    customSimbad = Simbad()
    # print(customSimbad.list_votable_fields())
    customSimbad.add_votable_fields(*fields)
    
    for i,id_source in enumerate(gaia_sources_result['source_id']):
        print('\nGAIA DR3 {0:d}'.format(id_source))
        if id_source == id_gaia_dr3:
            print('The {0:d}th source is actualy the target itself. '.format(id_source))
        else:
            query3 = """SELECT 
            EPOCH_PROP(gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.pmra,gaia_source.pmdec,gaia_source.radial_velocity,2015.5,{0:.6f}) 
            FROM gaiadr3.gaia_source 
            WHERE source_id = '{1:d}'""".format(date_obs.jyear,id_source)        
            j3 = Gaia.launch_job(query3)
            gaia_sources_prop_result = j3.get_results()        
            dico_contaminants['ID Gaia DR3'].append(id_source)
            delta_ra_deg = (gaia_sources_prop_result['EPOCH_PROP'][0][0]-gaia_prop_result['EPOCH_PROP'][0][0])*np.cos(np.deg2rad(gaia_prop_result['EPOCH_PROP'][0][1]))
            delta_dec_deg = gaia_sources_prop_result['EPOCH_PROP'][0][1]-gaia_prop_result['EPOCH_PROP'][0][1]
            delta_ra_arcsec = delta_ra_deg*3600
            delta_dec_arcsec = delta_dec_deg*3600
            dico_contaminants['RA (deg)'].append(gaia_sources_prop_result['EPOCH_PROP'][0][0])
            dico_contaminants['DEC (deg)'].append(gaia_sources_prop_result['EPOCH_PROP'][0][1])
            dico_contaminants['delta RA (arcsec)'].append(delta_ra_arcsec)
            dico_contaminants['delta DEC (arcsec)'].append(delta_dec_arcsec)
            
            # one can add here any keyword frrom the following list            
            # ['dist', 'solution_id', 'DESIGNATION', 'source_id', 'random_index', 'ref_epoch', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'parallax_over_error', 'pm', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr', 'astrometric_n_obs_al', 'astrometric_n_obs_ac', 'astrometric_n_good_obs_al', 'astrometric_n_bad_obs_al', 'astrometric_gof_al', 'astrometric_chi2_al', 'astrometric_excess_noise', 'astrometric_excess_noise_sig', 'astrometric_params_solved', 'astrometric_primary_flag', 'nu_eff_used_in_astrometry', 'pseudocolour', 'pseudocolour_error', 'ra_pseudocolour_corr', 'dec_pseudocolour_corr', 'parallax_pseudocolour_corr', 'pmra_pseudocolour_corr', 'pmdec_pseudocolour_corr', 'astrometric_matched_transits', 'visibility_periods_used', 'astrometric_sigma5d_max', 'matched_transits', 'new_matched_transits', 'matched_transits_removed', 'ipd_gof_harmonic_amplitude', 'ipd_gof_harmonic_phase', 'ipd_frac_multi_peak', 'ipd_frac_odd_win', 'ruwe', 'scan_direction_strength_k1', 'scan_direction_strength_k2', 'scan_direction_strength_k3', 'scan_direction_strength_k4', 'scan_direction_mean_k1', 'scan_direction_mean_k2', 'scan_direction_mean_k3', 'scan_direction_mean_k4', 'duplicated_source', 'phot_g_n_obs', 'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_g_mean_flux_over_error', 'phot_g_mean_mag', 'phot_bp_n_obs', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error', 'phot_bp_mean_flux_over_error', 'phot_bp_mean_mag', 'phot_rp_n_obs', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'phot_rp_mean_flux_over_error', 'phot_rp_mean_mag', 'phot_bp_rp_excess_factor', 'phot_bp_n_contaminated_transits', 'phot_bp_n_blended_transits', 'phot_rp_n_contaminated_transits', 'phot_rp_n_blended_transits', 'phot_proc_mode', 'bp_rp', 'bp_g', 'g_rp', 'radial_velocity', 'radial_velocity_error', 'rv_method_used', 'rv_nb_transits', 'rv_nb_deblended_transits', 'rv_visibility_periods_used', 'rv_expected_sig_to_noise', 'rv_renormalised_gof', 'rv_chisq_pvalue', 'rv_time_duration', 'rv_amplitude_robust', 'rv_template_teff', 'rv_template_logg', 'rv_template_fe_h', 'rv_atm_param_origin', 'vbroad', 'vbroad_error', 'vbroad_nb_transits', 'grvs_mag', 'grvs_mag_error', 'grvs_mag_nb_transits', 'rvs_spec_sig_to_noise', 'phot_variable_flag', 'l', 'b', 'ecl_lon', 'ecl_lat', 'in_qso_candidates', 'in_galaxy_candidates', 'non_single_star', 'has_xp_continuous', 'has_xp_sampled', 'has_rvs', 'has_epoch_photometry', 'has_epoch_rv', 'has_mcmc_gspphot', 'has_mcmc_msc', 'in_andromeda_survey', 'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy', 'classprob_dsc_combmod_star', 'teff_gspphot', 'teff_gspphot_lower', 'teff_gspphot_upper', 'logg_gspphot', 'logg_gspphot_lower', 'logg_gspphot_upper', 'mh_gspphot', 'mh_gspphot_lower', 'mh_gspphot_upper', 'distance_gspphot', 'distance_gspphot_lower', 'distance_gspphot_upper', 'azero_gspphot', 'azero_gspphot_lower', 'azero_gspphot_upper', 'ag_gspphot', 'ag_gspphot_lower', 'ag_gspphot_upper', 'ebpminrp_gspphot', 'ebpminrp_gspphot_lower', 'ebpminrp_gspphot_upper', 'libname_gspphot']            
            for key in ['pmra','pmdec','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','parallax']:
                if np.ma.is_masked(gaia_sources_result[key][i]):
                    dico_contaminants[key].append(np.nan)
                else:
                    dico_contaminants[key].append(gaia_sources_result[key][i])
            distance = np.sqrt(delta_ra_arcsec**2+delta_dec_arcsec**2)
            pa = np.rad2deg(np.arctan2(delta_ra_arcsec,delta_dec_arcsec))
            dico_contaminants['separation (arcsec)'].append(distance)
            dico_contaminants['PA (deg)'].append(pa)
    
            print('This companion is found at delta RA={0:.3f}arcsec and delta DEC={1:.3f}arcsec'.format(delta_ra_arcsec,delta_dec_arcsec))        
            print('Coresponding to a separation of {0:.3f} arcsec and a PA of {1:.2f} deg'.format(distance,pa))
            print('It has a G mag of {0:.1f}'.format(dico_contaminants['phot_g_mean_mag'][-1]))
            
            result_table_names = customSimbad.query_object('Gaia DR3 {0:d}'.format(id_source)) 
            try:
                for band in ['FLUX_J','FLUX_H','FLUX_K']:              
                    if not np.ma.is_masked(result_table_names[band][0]):
                        dico_contaminants[band].append(result_table_names[band][0])
                    else:
                        dico_contaminants[band].append(np.nan)
            except TypeError:
                for band in ['FLUX_J','FLUX_H','FLUX_K']:              
                        dico_contaminants[band].append(np.nan)            
    pd_contaminants = pd.DataFrame(dico_contaminants)
    pd_contaminants['distance'] = 1/(pd_contaminants['parallax']/1000.)
    return pd_contaminants
    
def save_results(pd_contaminants,path,basename='contaminants',img_size_px=1448):
    """
    Saves results in a csv and a reg file.

    Parameters
    ----------
    pd_contaminants : Panda DataFrame object
        DataFrame to store the properties of the contaminants 
    path: Pathlib.Path object
        the path where the dataframe will be saved as a csv file, along with the ref file
    img_size_px: int
        size of the image to match the reg file with the file you want to check.
        
    Returns
    -------
    None.

    """
    nb_contaminants_from_Gaia = len(pd_contaminants)
    if nb_contaminants_from_Gaia>0:
        pd_contaminants.to_csv(path.joinpath(basename+'.csv'))
        
        # then we create a ds9 arrow pointing at the source.
        center_px_coord = img_size_px//2+1 # +1 because DS9 is 1-based indexing
        reg_string = \
        """# Region file format: DS9 version 4.1
        global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
        image"""
        for j,sep in enumerate(pd_contaminants['separation (arcsec)']):
            pa_ds9 = pd_contaminants['PA (deg)'][j]+90
            sep_irdis = sep/0.01225
            reg_string = reg_string+"""
    # vector({0:d},{0:d},{1:.2f},{2:.2f}) vector=1""".format(center_px_coord,sep_irdis,pa_ds9)
        filename = path.joinpath('{0:s}_for_image_{1:d}x{1:d}.reg'.format(basename,img_size_px))
        txtfile = open(filename,'w')
        txtfile.write(reg_string)
        txtfile.close()   
    return
     
def search_for_gaia_contaminants(header=None,target=None, ra=None,dec=None,date=None,\
                          verbose=True,search_radius_arcsec=6.25,path='.'):
    """
    Retrieve the properties of the target and queries GAIA DR3 to find background sources
    in FoV
    Input: 
        - header: a fits header with all required keywords in it (optional if ra, dec and date are used)
        - target: a string for the target name
        - ra: a string representing the RA of the target in the format "02:27:01.06" (optional if header is used)
        - dec: a string representing the DEC of the target in the format "-12:47:00.06" (optional if header is used)
        - date_obs: astropy.time.Time object (optional if header is used)
        - verbose: bool to get verbose (optionnal, True by default)
    Output:
        - a dictionnary (see the function find_target_name_from_header.query_simbad
          for details on the content of this dictionnary)        
    """
    if header is None: # if no file is specified, the user must have specified a target name, date, ra and dec
        date_obs = Time(date)
        if (ra is None or dec is None) and target is not None: 
            coords = SkyCoord.from_name(target.strip())
        elif (ra is not None and dec is not None):
            coords = SkyCoord(ra+' '+dec, frame=ICRS, unit=(u.hourangle, u.deg))
        else:
            raise ValueError("If you don't specify a file, you must either give a target name or a set of RA/DEC coordinates.")
        simbad_dico = f.query_simbad(date_obs, coords, name=target,verbose=verbose)
    else: 
        header = fits.getheader(file)
        date_obs = Time(header['DATE-OBS'])
        simbad_dico = f.query_simbad_from_header(header,verbose=verbose)

    target_name = str(simbad_dico['simbad_MAIN_ID']).replace(' ','')
    date_iso = date_obs.datetime.date().isoformat()
    basename = 'contaminants_{0:s}_{1:s}_within_{2:.0f}_arcsec'.format(target_name,date_iso,search_radius_arcsec)  

    id_gaia_dr3 = get_gaia_dr3_id(simbad_dico,verbose=verbose)
    gaia_prop_result = propagate_pm_to_obs_date(id_gaia_dr3,date_obs,verbose=verbose)
    gaia_sources_result = query_gaia_sources(gaia_prop_result,search_radius_arcsec=search_radius_arcsec,verbose=verbose)
    pd_contaminants = propagat_pm_to_obs_date(gaia_sources_result,gaia_prop_result,id_gaia_dr3,date_obs,verbose=verbose)
    if path is not None:
        if type(path) == str:
            path = Path(path)
        save_results(pd_contaminants,path,basename=basename)
    return pd_contaminants
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Query Gaia to find the potential background contaminants in the field')
    parser.add_argument('-f','--file', type=str, help='fits file containing a header with the keyword DATE-OBS, RA and DEC')
    parser.add_argument('-t','--target', help='Target name (optional if a file is specified)',type=str)
    parser.add_argument('-d','--date', help='datetime in the format "2016-04-05 00:00:00" (optional if a file is specified)',type=str)
    parser.add_argument('--ra', help='RA in the format "02:27:01.06" (optional if a file is specified)',type=str)
    parser.add_argument('--dec', help='DEC in the format "-12:47:00.06" (optional if a file is specified)',type=str)
    parser.add_argument('-r','--radius', help='search radius in arcsec',type=float,default=12.25)
    parser.add_argument('-p','--path', help='path (as a string) to save results (optional,if not specified, the csv file and reg files are not saved)',type=str)
    parser.add_argument('-v', "--verbose", action="store_false",help='Display additional information')        
    parser.add_argument('-s', "--size", help='size of the image in px (optional, default is 1448px, corresponding the image size with the cADI reduction',type=int,default=1448)    
    
    args = parser.parse_args()
    file = args.file
    target=args.target
    date=args.date
    ra = args.ra
    dec=args.dec
    search_radius_arcsec=args.radius
    path = args.path
    verbose=args.verbose
    
    search_for_gaia_contaminants(header=file,target=target, ra=ra,dec=dec,date=date,\
                verbose=verbose,search_radius_arcsec=search_radius_arcsec,path=path)
    

