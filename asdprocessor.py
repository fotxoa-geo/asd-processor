import numpy as np
import asd_jump_correction
import asdreader
import scipy.interpolate as scinterp
import os
import fnmatch

# def get_closest_wvl_index(wvl, specific_wvl):
#     #Return the index of the wavelength nearest the specific wavelength
#     #specific_wvl is a single wavelength
#     idx = np.unravel_index(np.argmin(abs(wvl - specific_wvl)),wvl.shape)
#     return idx

def get_closest_wvl_index(wvl, specific_wvl):
    try:
        # Works for single input wavelengths that aren't in an np.array
        idx = np.argmin(abs(wvl - specific_wvl),axis = 0)
    except ValueError:
        #If specific_wvl is a 1D array of wavelengths to check, this will work
        if len(specific_wvl) > 1:
            if len(specific_wvl.shape) == 1:
                specific_wvl = specific_wvl[np.newaxis,:]
            if len(wvl.shape) == 1:
                wvl = wvl[:,np.newaxis]
        #Otherwise we'll error out again here
        idx = np.argmin(abs(wvl - specific_wvl),axis = 0)
    return idx

def cos_sim_to_ref(x_mat,y_ref):
    #x_mat is a n samples x m wavelength matrix
    #y_ref is a 1D length-m vector (signle spectrum)
    
    # Reorder x_mat so iterable direction is axis 0
    # *** Only deals with 2D x_mat for now
    if len(x_mat.shape) < 2:
        #Single spectrum
        np.expand_dims(x_mat,axis=0)
        
    wvl_ax = np.array(x_mat.shape) == len(y_ref)

    if wvl_ax[0]:
        # Wavelengths are in axis 0
        x_mat = np.transpose(x_mat)
        
    return np.dot(x_mat,y_ref)/(np.sqrt(np.sum(x_mat**2,axis=1))*np.sqrt(np.sum(y_ref**2)))

def get_filenames(data_dir, file_tag = '*'):
    return [f for f in os.listdir(data_dir) if fnmatch.fnmatch(f,file_tag)]

def get_asd_filetags(data_dir, file_tag = '*.asd', ext_length = 9):
    # This line first compiles a list of all files with file_ext in the data_dir, cutting off the last 9 elements
    # (When reading in asd files saved with a five digit number signifier, this will only leave the first file-tag)
    # *** Not the most adaptive way to do this, but should work for all asd files
    # *** Can update if this doesn't work for other people's files, though
    # list(dict.fromkeys( )) returns a list of the unique keys in the first list
    # sorted( ) then sorts the list for readability 
    
    # Convenience function for standard asd files
    
    filenames = get_filenames(data_dir, file_tag) #All .asd files in the directory
    file_tags = sorted(list(dict.fromkeys([f[:-ext_length].strip('_') for f in filenames]))) #Strip the number, find unique ones, and sort them
    return file_tags
  
def get_good_bands(bad_wvl_bnd, wavelengths):
    # bad_wl_bnd is an n x 2 matrix, where n is the number of bad bands
    # It identifies the first and last wavelength of the band in the same units as wavelengths
    # wavelengths is a vector of wavelengths
    # Returns boolean vector the same size as wavelengths where bad bands are set to False
    
    #Find indices of wavelengths
    bad_idx_bnd = get_closest_wvl_index(wavelengths,bad_wvl_bnd.flatten())
    bad_idx_bnd = np.reshape(bad_idx_bnd,bad_wvl_bnd.shape)
    
    #Set bad bands to false
    good_bands_idx = np.ones(wavelengths.shape).astype(bool)
    for ii in range(bad_idx_bnd.shape[0]):
        good_bands_idx[bad_idx_bnd[ii,0]:bad_idx_bnd[ii,1]] = False  
    return good_bands_idx

def get_mean_std_spectra(spec_mat,use_idx=None,keepdims=False):
    # Calculate final mean and std dev of many spectra, ignoring outliers etc
    # spec_mat is [n x m] where n is the number of measurements and m is the number of wavelengths
    
    if use_idx is None:
        use_idx = np.ones(spec_mat.shape[0]).astype(bool)
    
    mean    = np.mean(spec_mat[use_idx,:],axis=0,keepdims=keepdims)
    std_dev = np.std(spec_mat[use_idx,:],axis=0,keepdims=keepdims)
    return mean, std_dev
    
def calculate_outliers_std(spec_mat, good_bands_idx = None, outliers = None, sigma_factor=3):

    if outliers is not None:
        use_idx = np.logical_not(outliers)
    else:
        use_idx = None
        
    if good_bands_idx is None:
        good_bands_idx = np.ones(spec_mat.shape[1]).astype(bool)
        #Use all wavelengths if no mask is input

    # Reject reflectance outliers
    mean, std_dev = get_mean_std_spectra(spec_mat, use_idx = use_idx, keepdims=True)

    # Outliers found based on standard deviation
    # Only calculate outliers in good bands
    # *** Need to read in multiplier on standard deviation from user
    outliers_std = np.any(np.abs(spec_mat-mean)[:,good_bands_idx] > sigma_factor*std_dev[:,good_bands_idx],axis=1)
    return outliers_std

def calculate_outliers_cossim(spec_mat, good_bands_idx = None, outliers = None, threshold=0.985):

    if outliers is not None:
        use_idx = np.logical_not(outliers)
    else:
        use_idx = None
        
    if good_bands_idx is None:
        good_bands_idx = np.ones(spec_mat.shape[1]).astype(bool)
        #Use all wavelengths if no mask is input

    # Recalculate mean
    mean, std_dev = get_mean_std_spectra(spec_mat, use_idx = use_idx, keepdims=True)

    # Calculate outliers based on cosine similarity to mean
    cos_sim_mean = cos_sim_to_ref(spec_mat[:,good_bands_idx],np.squeeze(mean[:,good_bands_idx]))
    outliers_cos = cos_sim_mean < threshold # *** Arbitrary threshold need to tune ??
    return outliers_cos

def calculate_outliers(spec_mat, good_bands_idx = None, std_sigma_factor = 3, cossim_threshold = 0.985, outlier_list='all', outliers = None):
    #Convenience function for finding outliers from multiple methods
    
    if outlier_list == 'all':
        outlier_list = ['std','cossim']
    
    if outliers is None:
        #Initialize if we haven't already
        outliers = np.zeros(spec_mat.shape[0]).astype(bool)
    
    if 'std' in outlier_list: #outlier_type == 'all' or outlier_type == 'std':
        #Outliers based on standard deviation from mean
        outliers_std = calculate_outliers_std(spec_mat, good_bands_idx = good_bands_idx, outliers = outliers, sigma_factor = std_sigma_factor)
        outliers = np.logical_or(outliers,outliers_std)
       
    if 'cossim' in outlier_list: #outlier_type == 'all' or outlier_type == 'cossim':
        #Outliers based on deviation from cosine similarity to mean
        outliers_cos = calculate_outliers_cossim(spec_mat, good_bands_idx = good_bands_idx, outliers = outliers, threshold = cossim_threshold)
        outliers = np.logical_or(outliers,outliers_std)

    print(f'Total outliers: {np.sum(outliers)} (>{std_sigma_factor}x std dev: {np.sum(outliers_std)}, <{cossim_threshold} cos sim: {np.sum(outliers_cos)})')
    return outliers
    
class processor:
    def __init__(self, known_ref_idx = 0, flag_treat_REF_as_RAW=False, spectralon_data_path = './correction_data/spectralon_rfl.txt',\
                jump_corr_method = 'hueni', jump_corr_iterations = 5):
        #***What needs to be initialized?
        
        # Spectralon variables
        spectralon_data      = np.loadtxt(spectralon_data_path)
        self.spectralon_wl   = spectralon_data[100:,0]
        self.spectralon_rfl  = spectralon_data[100:,1]
        #assert np.all(self.wavelengths == self.spectralon_wl), 'Spectralon and ASD wavelengths must match!'
        
        self.spectralon_45deg_factor = 1.015

        # Water absorption bands
        # May not be bad bands strictly in this case, but still get higher variance here
        # so it seems unfair to use it to reject outliers
        # *** Need to read this in dynamically to allow user control
        self.bad_wvl_bnd = np.array([[1343, 1449], [1804, 1970]])
        
        self.flag_treat_REF_as_RAW = flag_treat_REF_as_RAW
        self.known_ref_idx         = known_ref_idx
        self.jump_corr_method      = jump_corr_method
        self.jump_corr_iterations  = jump_corr_iterations
        
        
    def load_asd_files(self,data_dir, file_tag):
        # Load all of the ASD spectrum and reference data associated with a particular file tag
        
        # Find asd files with the given file_tag in the given data_dir 
        filenames   = sorted(get_filenames(data_dir, file_tag + '*.asd'))
   
        self.n_file = len(filenames)

        # Read in first file to get metadata
        # Presumes that all files with same file_tag have the same metadata
        # (usually a safe assumption)
        print('Reading files, beginning with {}...'.format(data_dir + filenames[0]))
        asd_data        = asdreader.reader(data_dir + filenames[0])
        self.n_channels = asd_data.wavelengths.shape[0]
        self.wavelengths = asd_data.wavelengths.copy()
        self.good_bands_idx  = get_good_bands(self.bad_wvl_bnd, self.wavelengths)
        self.spectra_type    = asd_data.get_spectra_type()

        # Load in all asd_data for this file_tag
        self.spectra_mat   = np.zeros((self.n_file,self.n_channels))
        self.reference_mat = np.zeros((self.n_file,self.n_channels))
        for jj in range(self.n_file):
            # Read in data
            asd_data = asdreader.reader(data_dir + filenames[jj],verbose=False)
            # Store the spectra and reference
            self.spectra_mat[jj,]   = asd_data.spec
            self.reference_mat[jj,] = asd_data.reference 
            # asd_data.reference will be white reference if the data was taken in reflectance mode (REF)
            # asd_data.reference will be all zeros if the data was taken in raw mode (RAW)
            
        return self.spectra_mat, self.reference_mat
    
    def process_wr_REF(self):
        # Reflectance mode
        
        # Indices into overall matrics
        self.type_flag[:] = 1 #All are field spectra

        spectra_index = np.arange(self.spectra_mat.shape[0]) 
        
        #Find where reference changes from previous one
        ref_same      = np.hstack([False,np.all(np.diff(self.reference_mat,axis=0)==0,axis=1)]) #Include the first one
        ref_start_bnd = spectra_index[np.logical_not(ref_same)] #Beginning index for each unique reference
        
        self.ref_mat       = self.reference_mat[ref_start_bnd,:] #Unique spectra
        self.ref_i         = ref_start_bnd - 0.5
        self.ref_group     = np.arange(len(ref_start_bnd)) #What WR group each ref belongs to
        
    def process_wr_REF_as_RAW(self):
        
        #Identify reference spectra
        print(f'Using reference_mat as a known white reference spectra.')
        known_ref   = self.reference_mat[0,]
        cos_sim_vec = cos_sim_to_ref(self.spectra_mat,known_ref)

        ref_threshold = 0.9999
        ref_idx       = cos_sim_vec>=ref_threshold
        meas_idx      = cos_sim_vec<ref_threshold

        self.type_flag[meas_idx] = 1 #Field spectra
        self.type_flag[ref_idx]  = 0 #White reference
        
        spectra_index = np.arange(self.spectra_mat.shape[0]) 
        
        #Detected wr in spectra_mat
        self.ref_mat  = self.spectra_mat[ref_idx,:] #White references in spectra_mat
        self.ref_i    = spectra_index[ref_idx]
        
        #Insert white reference measurements from reference_mat
        ref_same      = np.hstack((False,np.all(np.diff(self.reference_mat,axis=0)==0,axis=1))) #Include the first one
        ref_start_bnd = spectra_index[np.logical_not(ref_same)] #Beginning index for each unique reference
        self.ref_mat  = np.vstack((self.ref_mat,self.reference_mat[ref_start_bnd,:])) #Unique spectra
        self.ref_i    = np.hstack((self.ref_i,ref_start_bnd - 0.5))
        
        #Sort so we group them correctly
        sort_idx      = np.argsort(self.ref_i)
        self.ref_i    = self.ref_i[sort_idx]
        self.ref_mat  = self.ref_mat[sort_idx,:]
        
        #Determine group index
        ref_same         = np.hstack((False,np.diff(self.ref_i,axis=0)<=1)) #Same group if 1 index or less away from each other
        ref_index     = np.arange(self.ref_i.shape[0])
        ref_start_bnd = ref_index[np.logical_not(ref_same)] #Beginning index for each unique reference
        ref_end_bnd   = np.hstack((ref_start_bnd[1:],len(self.ref_i)))
        self.ref_group = np.zeros_like(self.ref_i)
        for ii in range(ref_start_bnd.shape[0]):
            self.ref_group[ref_start_bnd[ii]:ref_end_bnd[ii]]=ii
        
    def process_wr_RAW(self):
        # Raw mode, where white reference standard measurements are interspersed among target measurements in spectra_mat

        print(f'Using index {self.known_ref_idx} as a known white reference spectra.')

        #Identify reference spectra
        known_ref   = self.spectra_mat[self.known_ref_idx,]
        cos_sim_vec = cos_sim_to_ref(self.spectra_mat,known_ref)

        ref_threshold = 0.9999
        ref_idx = cos_sim_vec>=ref_threshold
        meas_idx = cos_sim_vec<ref_threshold

        self.type_flag[meas_idx] = 1 #Field spectra
        self.type_flag[ref_idx]  = 0 #White reference
        
        spectra_index = np.arange(self.spectra_mat.shape[0]) 
        
        self.ref_mat  = self.spectra_mat[ref_idx,:] #Unique spectra in spectra_mat
        self.ref_i    = spectra_index[ref_idx]
        
        ref_same      = np.hstack((False,np.diff(self.ref_i,axis=0)<=1)) #Same group if 1 index or less away from each other
        ref_index     = np.arange(self.ref_i.shape[0])
        ref_start_bnd = ref_index[np.logical_not(ref_same)] #Beginning index for each unique reference
        ref_end_bnd   = np.hstack((ref_start_bnd[1:],len(self.ref_i)))
        self.ref_group = np.zeros_like(self.ref_i)
        for ii in range(ref_start_bnd.shape[0]):
            self.ref_group[ref_start_bnd[ii]:ref_end_bnd[ii]]=ii
        
        #self.ref_group   = np.arange(np.sum(np.logical_not(ref_same)))
    
    def identify_white_references(self):

        #Intialize type flag
        #0 = white reference; 1 = field spectra; -1 = outlier; -2 = wref outlier
        self.type_flag = np.zeros(self.n_file)
        
        #Sort spectra and reference into their own matrixes, with index into "time", and group index for references
        if self.spectra_type == 'RAW':
            self.process_wr_RAW()
        elif self.spectra_type == 'REF':
            if self.flag_treat_REF_as_RAW:
                self.process_wr_REF_as_RAW()
            else:
                self.process_wr_REF()
        else:
            print(f'Data captured in mode {self.spectra_type} not currently supported. Exiting...')
            # *** Should probably throw an error instead
            quit()
        
        print(f'Number of unique white reference groups detected: {np.max(self.ref_group)+1}')
        
        # Populates meas_i, ref_mat, ref_i, ref_group fields
        # Identify white reference outliers
        ref_outliers = calculate_outliers(self.ref_mat, good_bands_idx = self.good_bands_idx)

        # Update type_flag and reference mats for outliers
        valid_ref_i   = np.logical_and(np.mod(self.ref_i,1)==0,ref_outliers) #Only outliers that are in spectra_mat (RAW or treat_REF_as_RAW)
        
        if len(self.ref_i[valid_ref_i])>0:
            self.type_flag[self.ref_i[valid_ref_i]] = -2
                
        #Delete outliers from ref_mat, ref_i, ref_group fields
        self.ref_mat      = np.delete(self.ref_mat, ref_outliers, axis = 0)
        self.ref_i        = np.delete(self.ref_i, ref_outliers)
        self.ref_group    = np.delete(self.ref_group, ref_outliers)
        return self.ref_mat, self.ref_i, self.ref_group
    
    def interpolate_white_references(self, interp_indices, interp_method = 'linear'):
                
        # Get average, std of each group
        ref_group_unique  = np.unique(self.ref_group)
        n_groups          = len(ref_group_unique)
        self.ref_mean_mat = np.zeros((n_groups,self.ref_mat.shape[1]))
        self.ref_std_mat  = np.zeros((n_groups,self.ref_mat.shape[1]))
        self.ref_mean_i   = np.zeros(n_groups)
        for ii in range(n_groups):
            use_idx = self.ref_group == ref_group_unique[ii]
            mean, std_dev = get_mean_std_spectra(self.ref_mat, use_idx = use_idx)
            
            self.ref_mean_mat[ii,:] = mean
            self.ref_std_mat[ii,:]  = std_dev
            self.ref_mean_i[ii]     = np.mean(self.ref_i[use_idx]) #Mean time stamp
        
        #Value to fill with if we have spectra before / after the white reference range
        fill_value = (self.ref_mean_mat[0,:],self.ref_mean_mat[-1,:])
        
        # Create interpolator
        #Interpolate in 1D down the references, with ref_mean_i as the "time" axis
        ref_func = scinterp.interp1d(self.ref_mean_i, self.ref_mean_mat, kind = interp_method, axis = 0,\
                                    bounds_error = False, fill_value = fill_value)
        
        # Apply interpolator
        #Return interpolated white references at the "time points" of the measurements
        ref_interp_mat = ref_func(interp_indices)
        return ref_interp_mat
            
    def process_white_references(self,interp_method = 'linear'):
                
        self.identify_white_references()
        
        spectra_index  = np.arange(self.spectra_mat.shape[0]) 
        meas_i         = spectra_index[self.type_flag == 1] #Index of measurements
        ref_group_unique  = np.unique(self.ref_group)
        n_groups          = len(ref_group_unique)
        if n_groups>1:
            #If we have more than one white reference group to work with
            ref_interp_mat = self.interpolate_white_references(meas_i, interp_method = interp_method)
        else:
            ref_interp_mat = np.zeros((len(meas_i),self.spectra_mat.shape[1]))
            ref_interp_mat[:,]=self.ref_mat[0,:]
                   
        # Put the interpolated reference spectra into the reference_mat at the measurement locations
        # self.meas_i is also the index into spectra_mat and reference_mat
        self.reference_mat[meas_i,:] = ref_interp_mat
        
    def calculate_reflectances(self):
        #Take the ratio to get reflectance at valid measurement locations
        meas_idx = self.type_flag == 1
        self.reflectance_mat = self.spectra_mat/self.reference_mat
        return self.reflectance_mat[meas_idx,:]

    def apply_spectralon_rfl_corr(self):
        #Account for spectralon reflectance 
        self.reflectance_mat = self.spectralon_rfl*self.reflectance_mat
        meas_idx = self.type_flag == 1
        return self.reflectance_mat[meas_idx,:]

    def apply_spectralon_45deg_corr(self):
        #Account for spectralon being at 45 deg to measurement angle
        self.reflectance_mat = self.spectralon_45deg_factor*self.reflectance_mat
        meas_idx = self.type_flag == 1
        return self.reflectance_mat[meas_idx,:]
    
    def apply_jump_correction(self,jump_corr_method=None,jump_corr_iterations=None,asd_coeff_path='./ASD_Jump_Correction/ASD_Jump_Correction/asd_temp_corr_coeffs.mat'):
        meas_idx = self.type_flag == 1
        if jump_corr_method is None:
            jump_corr_method = self.jump_corr_method
        if jump_corr_iterations is None:
            jump_corr_iterations = self.jump_corr_iterations
            
        print(jump_corr_method, jump_corr_iterations)
        self.reflectance_mat[meas_idx,:] = asd_jump_correction.apply_jump_correction(self.reflectance_mat[meas_idx,:], self.wavelengths,jump_corr_method = jump_corr_method, iterations=jump_corr_iterations, asd_coeff_path = asd_coeff_path)   
        return self.reflectance_mat[meas_idx,:]
    
    def apply_corrections(self,corr_list = 'all',jump_corr_method=None,jump_corr_iterations=None):
        #Convenience function to call all the corrections
        if corr_list == 'all':
            corr_list = ['spec_rfl','spec_45deg','jump_corr']
        
        if 'spec_rfl' in corr_list:
            self.apply_spectralon_rfl_corr()
        if 'spec_45deg' in corr_list:
            self.apply_spectralon_45deg_corr()
        if 'jump_corr' in corr_list:
            self.apply_jump_correction(jump_corr_method=jump_corr_method,jump_corr_iterations=jump_corr_iterations)
        return self.reflectance_mat[meas_idx,:]
        
    def calculate_outliers(self, std_sigma_factor = 3, cossim_threshold = 0.985):
        meas_idx = np.where(self.type_flag == 1)[0]
        #Only calculate outliers for valid measurements
        outliers = calculate_outliers(self.reflectance_mat[meas_idx,:], good_bands_idx = self.good_bands_idx, std_sigma_factor = std_sigma_factor, cossim_threshold = cossim_threshold)
        outlier_idx = meas_idx[outliers]
        self.type_flag[outlier_idx] = -1 #Mark outliers
        return outliers
            
    def calculate_final_reflectance(self):
        meas_idx = self.type_flag == 1
        mean, std_dev = get_mean_std_spectra(self.reflectance_mat, use_idx = meas_idx)
        self.rfl_mean = mean
        self.rfl_std  = std_dev
        self.rfl_mat  = self.reflectance_mat[meas_idx,:]
        return self.rfl_mean, self.rfl_std, self.rfl_mat
    
    def process_asd_files(self,data_dir, file_tag):
         # Read in all files associated with a certain file_tag
        self.load_asd_files(data_dir, file_tag)
        
        # Identify white references, identify outliers, interpolate between them if desired
        self.process_white_references() 
        
        # Reflectance based on white references in step above
        self.calculate_reflectances()
         
        #Account for spectralon reflectance, 
        #Sun being at 45 deg to measurement angle for white reference
        #ASD jumps at 1000 and 1800 nm
        self.apply_corrections()
        
        #Find reflectance outliers
        self.calculate_outliers()
        
        #Get final mean, std
        self.calculate_final_reflectance()
        return self.rfl_mean, self.rfl_std, self.rfl_mat