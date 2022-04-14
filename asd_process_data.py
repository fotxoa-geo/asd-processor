import numpy as np
import os
import sys
import fnmatch
import argparse
import matplotlib.pyplot as plt
import scipy.io as scio
import asdreader
from asd_jump_correction import asd_jump_correction

import matplotlib as mpl
from cycler import cycler

parser = argparse.ArgumentParser(description="Read in reflectance ASD data.")

parser.add_argument('--base_dir', type=str, default='/Users/reckert/Documents/Data/SHIFT_2022_Field/')
parser.add_argument('--data_dir', type=str, default='20220301')
parser.add_argument('--out_dir', type=str, default='output')
parser.add_argument('--out_tag', type=str, default='_all_spectra')
parser.add_argument('--save_dir', type=str, default='output/all/')
parser.add_argument('--save_file', type=str, default='shift_field_data_v1.mat')
parser.add_argument('--file_ext', type=str, default='.asd')
# parser.add_argument('--file_tags', type=str, default=None)
parser.add_argument('--file_tags', type=str, nargs='+', default=None)
parser.add_argument('--known_ref_idx', type=int, default=0) #Raw mode
parser.add_argument('--special_file_tags', type=str, nargs='+', default=None)
parser.add_argument('--special_known_idx', type=int, nargs='+', default=None)
parser.add_argument('--nosave_file_tags', type=str, nargs='+', default=None)
parser.add_argument('--flag_no_plots', action='store_true')
parser.add_argument('--flag_save_individ', action='store_true')

#Jump correction controls
parser.add_argument('--flag_interpolate_H2O', action='store_true')
parser.add_argument('--jump_corr_iterations', type=int, default=5)
parser.add_argument('--jump_corr_method', type=str, default='asdparabolic')   #'asdparabolic' or 'hueni'
parser.add_argument('--negatives_corr_method', type=str, default='parabolic') #'parabolic' or 'offset'
parser.add_argument('--interpolate_H2O_method', type=str, default='parabolic') #'parabolic' or 'smooth'

args = parser.parse_args()

# def slice-jump-correction(stuff):
# *** Need to step-correct data!
# A few sources, potentially:
# A. Hueni and A. Bialek, "Cause, Effect, and Correction of Field Spectroradiometer Interchannel Radiometric Steps," 
# in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 10, no. 4, pp. 1542-1551, 
# April 2017, doi: 10.1109/JSTARS.2016.2625043.
# (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7819458)
# --> Implemented: https://www.mathworks.com/matlabcentral/fileexchange/57569-asd-full-range-spectroradiometer-jump-correction
# Beal, D. and Eamon, M., 1996. Dynamic, Parabolic Linear Transformations of 'Stepped' Radiometric Data. 
# Analytical Spectral Devices Inc., Boulder, CO. (Could not find the paper itself!)
# --> Implemented: https://rdrr.io/cran/spectacles/man/splice.html
# --> Implemented in Mark Helmlinger's code
# return step-corrected data

#Cosine similarity of a matrix to a single reference
def cos_sim_to_ref(x_mat,y_ref):
    return np.dot(x_mat,y_ref)/(np.sqrt(np.sum(x_mat**2,axis=1))*np.sqrt(np.sum(y_ref**2)))

def main(args): 
    
    # Set up directories
    # *** Need to make this adaptive to Windows vs. Linux
    data_dir = args.base_dir + args.data_dir + '/' 
    out_dir  = args.base_dir + args.out_dir + '/' + args.data_dir + '/'
    fig_dir  = args.base_dir + args.out_dir + '/figures/' + args.data_dir + '/'
    save_dir = args.base_dir + args.save_dir
    
    spectralon_data = np.loadtxt(args.base_dir + 'metadata/spectralon_rfl.txt')
    spectralon_45deg_factor = 1.015
    spectralon_wl   = spectralon_data[100:,0]
    spectralon_rfl  = spectralon_data[100:,1]
    
    # Load coefficients matrix for (Hueni method) jump correction
    coeff_data = scio.loadmat('./ASD_Jump_Correction/ASD_Jump_Correction/asd_temp_corr_coeffs.mat')
    asd_coeffs = coeff_data['asd_temp_corr_coeffs'] #Checked, and is loading in correct direction (or should be)
    # Note: Coeffs file not included in Github due to memory considerations.
    # Please see https://www.mathworks.com/matlabcentral/fileexchange/57569-asd-full-range-spectroradiometer-jump-correction to download the data, along with A. Hueni and A. Bialek's original matlab code

    # Make the output directory if it doesn't exist
    print('Output directory: {}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Output directory: {}'.format(fig_dir))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    print('Output directory: {}'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.file_tags is None:
        #Did not specify a list of file tags to process, so will instead assume we want to process everything in the directory
        file_tag_list = sorted(list(dict.fromkeys([f[:-9].strip('_') for f in os.listdir(data_dir) if fnmatch.fnmatch(f,'*' + args.file_ext)])))
        # This line first compiles a list of all files with file_ext in the data_dir, cutting off the last 9 elements
        # (When reading in asd files saved with a five digit number signifier, this will only leave the first file-tag)
        # *** Not the most adaptive way to do this, but should work for all asd files
        # *** Can update if this doesn't work for other people's files, though
        # list(dict.fromkeys( )) returns a list of the unique keys in the first list
        # sorted( ) then sorts the list for readability     
    else:
        file_tag_list = args.file_tags
        
    if not args.flag_save_individ:
        #Load overall data
        data_all = scio.loadmat(save_dir + args.save_file, squeeze_me = True)
    
    print(f'Processing {args.file_ext} files begining with tags:\n{file_tag_list}')
    n_spectra_used = np.zeros(len(file_tag_list))
    n_outliers = n_spectra_used.copy()
    for ii in range(len(file_tag_list)):
        # List all files in data_dir with given file_tag
        file_tag = file_tag_list[ii]
        print(file_tag)
        filenameL = sorted([f for f in os.listdir(data_dir) if fnmatch.fnmatch(f, file_tag + '*' + args.file_ext)])    
        n_file = len(filenameL)

        # Read in first file to get metadata
        # Presumes that all files with same file_tag have the same metadata
        # (usually a safe assumption)
        print('Reading files, beginning with {}...'.format(data_dir + filenameL[0]))
        asd_data = asdreader.reader(data_dir + filenameL[0])
        n_channels = asd_data.wavelengths.shape[0]
        
        if ii == 0:
            mean_spectra_mat = np.zeros((len(file_tag_list),n_channels))
            std_spectra_mat = np.zeros((len(file_tag_list),n_channels))
            asd_wavelengths = asd_data.wavelengths.copy()
            
            assert np.all(asd_wavelengths == spectralon_wl), 'Spectralon and ASD wavelengths must match!'
            

        # Load in all asd_data for this file_tag
        spectra_mat = np.zeros((n_file,n_channels))
        reference_mat = spectra_mat.copy()
        for jj in range(n_file):
            # Read in data
            asd_data = asdreader.reader(data_dir + filenameL[jj],verbose=False)
            # Store the spectra and reference
            spectra_mat[jj,]   = asd_data.spec
            reference_mat[jj,] = asd_data.reference 
            # asd_data.reference will be white reference if the data was taken in reflectance mode (REF)
            # asd_data.reference will be all zeros if the data was taken in raw mode (RAW)
        
        if asd_data.get_spectra_type() == 'REF':
            # Reflectance mode
            
            reflectance_mat = spectra_mat/reference_mat
                
        elif asd_data.get_spectra_type() == 'RAW':
            # Raw mode, where white reference standard measurements are interspersed among target measurements in spectra_mat
            
            print(f'Using index {args.known_ref_idx} as a known white reference spectra.')
            
            #Identify reference spectra
            # *** Could identify in another way, but here I am doing the simple thing of feeding in a known index
            known_ref = spectra_mat[args.known_ref_idx,]
            cos_sim_vec = cos_sim_to_ref(spectra_mat,known_ref)

            spectra_index = np.arange(spectra_mat.shape[0]) #*Could also load in datetime here from .asd files

            ref_threshold = 0.999
            ref_idx = cos_sim_vec>=ref_threshold
            meas_idx = cos_sim_vec<ref_threshold
            
            #Find where reference spectra begin and end
            ref_change = np.hstack([0,np.diff(ref_idx.astype(int))])
            ref_begin_bnd = spectra_index[ref_change>0]
            ref_end_bnd = spectra_index[ref_change<0]

            #Case where we start with reference spectra
            try:
                ref_begin_bnd = np.hstack([0, ref_begin_bnd]) if ref_begin_bnd[0]>ref_end_bnd[0] else ref_begin_bnd
            except IndexError:
                ref_begin_bnd = np.array([0]) #Case where there's only one set of reference spectra at the beginning
                
                
            #Case where we end with reference spectra
            ref_end_bnd = np.hstack([ref_end_bnd, spectra_index[-1]]) if ref_begin_bnd[-1]>ref_end_bnd[-1] else ref_end_bnd
            
            
            #Get average and standard deviation for reference groups
            ref_mean = np.zeros((len(ref_begin_bnd),spectra_mat.shape[1]))
            ref_std  = ref_mean.copy()
            
            ref_nspectra = np.zeros(len(ref_begin_bnd))
            
            ref_groups_outlier = [] #Outliers within groups
            ref_groups = [] #Group reference spectra
            for kk in range(len(ref_begin_bnd)):
                ref_group = spectra_mat[ref_begin_bnd[kk]:ref_end_bnd[kk],:]
                
                # Reject outliers within group
                ref_outliers = np.any(np.abs(ref_group-np.mean(ref_group,axis=0)) > 3*np.std(ref_group,axis=0),axis=1)
                
                # Get mean and standard deviation of each group
                ref_mean[kk,:] = np.mean(ref_group[np.logical_not(ref_outliers),:],axis=0)
                ref_std[kk,:]  = np.std(ref_group[np.logical_not(ref_outliers),:],axis=0)
                
                #Save group and outlier indices (unclear if we'll use this at all)
                ref_groups.append(ref_group) #Append full set of spectra in a single group
                ref_groups_outlier.append(ref_outliers)
                
                ref_nspectra[kk] = np.sum(np.logical_not(ref_outliers)) #Number of spectra in mean, std
                 
            
            # #Potential output plot
            # plt.figure(figsize=(8,4))
            # plt.subplot(1,2,1)
            # plt.plot(asd_data.wavelengths,np.transpose(ref_mean))
            # plt.title('Mean reference')
            # plt.subplot(1,2,2)
            # plt.plot(asd_data.wavelengths,np.transpose(ref_std))
            # plt.title('Standard deviation of reference')
            
            #Only keep the reference spectra that 
            ref_good_idx = np.logical_and(np.all(ref_std<1000,axis=1),ref_nspectra>1) #The standard deviation of all wavelengths is low, and the number of spectra is more than 1 (otherwise the standard deviation is meaningless)
            ref_good_bnd = np.vstack((ref_begin_bnd[ref_good_idx],ref_end_bnd[ref_good_idx])) #New bounds, if we've rejected any groups of references
            ref_good_index = np.arange(len(ref_begin_bnd))[ref_good_idx] # Index into mean, std, and group arrays
            
            # *** Could do a much fancier interpolation scheme between the different reference groups
            # For now, just identify which reference is closest and use that
            ref_good_bnd_flat = ref_good_bnd.flatten()
            ref_good_index_flat = np.vstack((ref_good_index,ref_good_index)).flatten() #To echo the flattening process for the bounds

            #Find the closest reference group for each measurement
            meas_ref_index = ref_good_index_flat[np.argmin(np.abs(np.expand_dims(spectra_index[meas_idx],axis=1) - np.expand_dims(ref_good_bnd_flat,axis=0)),axis=1)] #Index into mean, std, and group arrays
            
            #Take the ratio to get reflectance
            reflectance_mat = spectra_mat[meas_idx,:]/ref_mean[meas_ref_index,:]
            
        else:
            print(f'Data captured in mode {asd_data.get_spectra_type()} not currently supported. Exiting...')
            # *** Should probably throw an error instead
            quit()
         
        #Account for spectralon reflectance and 45 degree BRDF effect
        reflectance_mat = spectralon_45deg_factor*spectralon_rfl*reflectance_mat
        
        #Spectral jump correction at 1000-1001 nm and 1800-1801 nm
        for jj in range(reflectance_mat.shape[0]):
            corrected_spectrum, outside_T, spec_corr_factors, jump_size_matrix, processing_notes = asd_jump_correction(asd_coeffs, reflectance_mat[jj,:], asd_wavelengths, interpolate_H2O=args.flag_interpolate_H2O, iterations=args.jump_corr_iterations, jump_corr_method = args.jump_corr_method, negatives_corr_method=args.negatives_corr_method, interpolate_H2O_method=args.interpolate_H2O_method)
            reflectance_mat[jj,:] = corrected_spectrum.copy()
            
        corr_tag = f'_corr_s_j_{args.jump_corr_method}'
        corr_title = f'spectralon and jump corrected ({args.jump_corr_method})'
        
        plt.figure()
        plt.plot(asd_data.wavelengths,np.transpose(reflectance_mat))
        plt.title('All reflectance spectra\n' + corr_title)
        
        # Reject reflectance outliers
        mean    = np.mean(reflectance_mat,axis = 0,keepdims=True)
        std_dev = np.std(reflectance_mat,axis=0,keepdims=True)
        
        # Water absorption bands
        # May not be bad bands strictly in this case, but still get higher variance here
        # so it seems unfair to use it to reject outliers
        # *** Need to read this in dynamically to allow user control
        bad_wl_bnd = np.array([[1343, 1449,1804, 1970]])
        bad_idx_bnd = np.argmin(np.abs(asd_data.wavelengths[:,np.newaxis]-bad_wl_bnd),axis=0)
        bad_bands = np.zeros(asd_data.wavelengths.shape).astype(bool)
        bad_bands[bad_idx_bnd[0]:bad_idx_bnd[1]]=True
        bad_bands[bad_idx_bnd[2]:bad_idx_bnd[3]]=True
        good_bands = np.logical_not(bad_bands)
        
        # Outliers found based on standard deviation
        # Only calculate outliers in good bands
        # *** Need to read in multiplier on standard deviation from user
        outliers_std = np.any(np.abs(reflectance_mat-mean)[:,good_bands] > 3*std_dev[:,good_bands],axis=1)
        
        # Recalculate mean
        mean    = np.mean(reflectance_mat[np.logical_not(outliers_std),:],axis = 0,keepdims=True)
        
        # Could also calculate outliers based on cosine similarity to mean?
        # *** Recalculate mean, std dev before doing this?
        cos_sim_mean = cos_sim_to_ref(reflectance_mat[:,good_bands],np.squeeze(mean[:,good_bands]))
        outliers_cos = cos_sim_mean < 0.985 # *** Arbitrary threshold need to tune ??
        
        # Final outlier array
        outliers = np.logical_or(outliers_std,outliers_cos)
        print(f'Total outliers: {np.sum(outliers)} (>3x std dev: {np.sum(outliers_std)}, <0.985 cos sim: {np.sum(outliers_cos)})')
        
        if args.special_file_tags is not None and file_tag in args.special_file_tags:
            known_idx = args.special_known_idx[args.special_file_tags.index(file_tag)]
            cos_sim_known = cos_sim_to_ref(reflectance_mat[:,good_bands],np.squeeze(reflectance_mat[known_idx,good_bands]))
            outliers = cos_sim_known < 0.99
            print(f'Special case! Using known reference index {known_idx}, outliers: {np.sum(outliers)}')
            
        # Calculate final mean and std dev of reflectance
        mean    = np.mean(reflectance_mat[np.logical_not(outliers),:],axis=0)
        std_dev = np.std(reflectance_mat[np.logical_not(outliers),:],axis=0)
        
        print(f'Mean value: {np.mean(mean):.3f}, Mean std dev: {np.mean(std_dev):.3f},Percent variation: {100*np.mean(std_dev)/np.mean(mean):.3f}%')
        mean_spectra_mat[ii,] = mean
        std_spectra_mat[ii,]  = std_dev
        n_spectra_used[ii]    = np.sum(np.logical_not(outliers))
        n_outliers[ii]        = np.sum(outliers)
        
        #Get site info
        site_tag = filenameL[0].split('_')[1] #Second part of name
        flag_nosave = (args.nosave_file_tags is not None and site_tag in args.nosave_file_tags)
        if not args.flag_save_individ and not flag_nosave:
            #Save in overall matrix
            site_tag = site_tag.split('-')[0] #Take first number if there's a dash
            
            site_num = int(site_tag.strip('Site').strip('site')) #Convert to a number
            print(site_num)
            
            date_tag = args.data_dir 
            
            #Grow the matrices if needed
            if not np.any(site_num == data_all['site_label']):
                print(f'Adding site {site_num} to overall data matrices')
                data_all['site_label']       = np.hstack((data_all['site_label'],np.array([site_num])))
                data_all['mean_spectra_mat'] = np.vstack((data_all['mean_spectra_mat'],np.zeros((1,data_all['mean_spectra_mat'].shape[1],data_all['mean_spectra_mat'].shape[2]))))
                data_all['std_spectra_mat'] = np.vstack((data_all['std_spectra_mat'],np.zeros((1,data_all['std_spectra_mat'].shape[1],data_all['std_spectra_mat'].shape[2]))))
                data_all['n_spectra_used'] = np.vstack((data_all['n_spectra_used'],np.zeros((1,data_all['n_spectra_used'].shape[1]))))   
                data_all['n_outliers'] = np.vstack((data_all['n_outliers'],np.zeros((1,data_all['n_outliers'].shape[1]))))  
            
            if not np.any(date_tag == data_all['date_label']):
                print(f'Adding date {date_tag} to overall data matrices')
                data_all['date_label']       = np.hstack((data_all['date_label'],np.array([date_tag])))
                data_all['mean_spectra_mat'] = np.hstack((data_all['mean_spectra_mat'],np.zeros((data_all['mean_spectra_mat'].shape[0],1,data_all['mean_spectra_mat'].shape[2]))))
                data_all['std_spectra_mat'] = np.hstack((data_all['std_spectra_mat'],np.zeros((data_all['std_spectra_mat'].shape[0],1,data_all['std_spectra_mat'].shape[2]))))
                data_all['n_spectra_used'] = np.hstack((data_all['n_spectra_used'],np.zeros((data_all['n_spectra_used'].shape[0],1))))   
                data_all['n_outliers'] = np.hstack((data_all['n_outliers'],np.zeros((data_all['n_outliers'].shape[0],1)))) 
                
            site_idx = np.where(site_num == data_all['site_label'])[0]
            date_idx = np.where(date_tag == data_all['date_label'])[0]
            
            print(f'Site tag: {site_tag} (num {site_num}); site idx {site_idx}, date idx {date_idx}')
            
            data_all['mean_spectra_mat'][site_idx,date_idx,:] = mean
            data_all['std_spectra_mat'][site_idx,date_idx,:] = std_dev
            data_all['n_spectra_used'][site_idx,date_idx] = n_spectra_used[ii] 
            data_all['n_outliers'][site_idx,date_idx] = n_outliers[ii] 
                
                
        
        if not args.flag_no_plots:
            mpl.rcParams['axes.prop_cycle'] = cycler(color=mpl.cm.rainbow(np.linspace(0,1,num=reflectance_mat.shape[0])))

            line_width = 3
            plt.figure()
            for jj in range(reflectance_mat.shape[0]):
                line_style = '--' if outliers[jj] else '-'
                plt.plot(asd_data.wavelengths,reflectance_mat[jj,:],line_style,linewidth=line_width/2)
            plt.plot(asd_data.wavelengths,np.squeeze(mean),'k',linewidth=line_width)
            # plt.plot(asd_data.wavelengths,np.squeeze(mean-std_dev),'0.2',linewidth=line_width/2,linestyle='-.')     
            # plt.plot(asd_data.wavelengths,np.squeeze(mean+std_dev),'0.2',linewidth=line_width/2,linestyle='-.') 
            plt.plot(asd_data.wavelengths,np.squeeze(mean-2*std_dev),'0.4',linewidth=line_width/2,linestyle='-.')     
            plt.plot(asd_data.wavelengths,np.squeeze(mean+2*std_dev),'0.4',linewidth=line_width/2,linestyle='-.') 
            plt.title(f'{file_tag}{corr_tag}: All spectra (outliers in --)\nFinal mean (black), 2x Std Dev (gray, -.)')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.savefig(fig_dir + 'allSpectra_' + file_tag + '_' + args.data_dir + corr_tag + args.out_tag + '.png',bbox_inches='tight')
            # *** Saving convention? .mat file? .txt file? Aggregate (all file_tags together) or separated (each with own file)?
            # Can also save info like: number of spectra averaged, number of unique reference spectra, number of outliers
            plt.close('all')

        
    if not args.flag_no_plots:
        mpl.rcParams['axes.prop_cycle'] = cycler(color=mpl.cm.rainbow(np.linspace(0,1,num=len(file_tag_list))))
        plt.figure()
        plt.plot(asd_data.wavelengths,np.transpose(mean_spectra_mat),linewidth=line_width)
        plt.legend(file_tag_list)
        plt.plot(asd_data.wavelengths,np.transpose(mean_spectra_mat-2*std_spectra_mat ),linewidth=line_width/2,linestyle='-.',alpha=0.5)     
        plt.plot(asd_data.wavelengths,np.transpose(mean_spectra_mat+2*std_spectra_mat),linewidth=line_width/2,linestyle='-.',alpha=0.5) 
        plt.savefig(fig_dir + 'mean_2xstd_all_' + args.data_dir + corr_tag + args.out_tag + '.png',bbox_inches='tight')
        plt.close('all')

#     plt.figure()
#     for ii in range(len(file_tag_list)):
#         plt.plot(asd_data.wavelengths,np.squeeze(mean_spectra_mat[ii,]),linewidth=line_width,color=color_mat[ii,])
        
#         plt.plot(asd_data.wavelengths,np.squeeze(mean_spectra_mat[ii,]-2*std_spectra_mat[ii,] ),linewidth=line_width/2,linestyle='-.',color=color_mat[ii,],alpha=0.5)     
#         plt.plot(asd_data.wavelengths,np.squeeze(mean_spectra_mat[ii,]+2*std_spectra_mat[ii,] ),linewidth=line_width/2,linestyle='-.',color=color_mat[ii,],alpha=0.5) 
#     plt.savefig(fig_dir + 'mean_2xstd_all_' + args.data_dir + args.out_tag + '.png',bbox_inches='tight')
#     plt.close('all')
    
    if args.flag_save_individ:
        mdict = {'mean_spectra_mat': mean_spectra_mat,
                 'std_spectra_mat':  std_spectra_mat,
                 'file_tag_list':    file_tag_list,
                 'n_spectra_used':   n_spectra_used,
                 'n_outliers':       n_outliers,
                 'corr_title':       corr_title}

        scio.savemat(out_dir + args.data_dir + corr_tag + args.out_tag + '.mat',mdict)
    else:
        scio.savemat(save_dir + args.save_file,data_all)    
    
    
        

if __name__ == '__main__':
    main(args)
