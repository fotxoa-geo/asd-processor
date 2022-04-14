# ASD spectroradiometer jump correction
# Adapted from Matlab code from Hueni, A. and Bialek, A. available at https://www.mathworks.com/matlabcentral/fileexchange/57569-asd-full-range-spectroradiometer-jump-correction
# Code adapted to Python by Regina Eckert at NASA's Jet Propulsion Laboratory (reckert@jpl.nasa.gov)
# Initial version: 4/12/2022

# %   ASD Full Range Spectroradiometer Jump Correction
# %   ------------------------------------------------
# %   
# %   Uses empirical correction coefficients to correct for temperature
# %   related radiometric inter-channel steps.
# %   For more information please see: 
# %   Hueni, A. and Bialek, A. (2017). "Cause, Effect and Correction of Field Spectroradiometer Inter-channel Radiometric Steps." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 10(4): 1542-1551.
# %
# %   Please cite the above paper if you use this method in any published work.
# %
# %   ------------------------------------------------
# %   (c) 2016-2021 A.Hueni, RSL, University of Zurich
# %   ------------------------------------------------


import numpy as np

def get_closest_wvl_index(wvl, specific_wvl):
    idx = np.unravel_index(np.argmin(abs(wvl - specific_wvl)),wvl.shape)
    return idx

def solve_quadratic_temperature(a,b,c):
    Ts = np.zeros(2)
    T_solution = True
    
    # Roots to quadratic equation
    Ts[0] = (-b + np.sqrt(b**2 - 4 * a * c))/(2*a)
    Ts[1] = (-b - np.sqrt(b**2 - 4 * a * c))/(2*a)
    
    # % temperature range: -10C - 70C : some sunglint effects can lead to
    # % jump sizes that result in unreasonalbly high assumed temperatures:
    # % these effects are essentially NOT temperature effects but field of
    # % view issues!
    
    if np.all(np.isreal(Ts)):
        # print('All real Ts:')
        # print(Ts)
        #If both roots are real
        T_feasible_ind = np.logical_and(Ts > -10,Ts < 70)
        if np.all(T_feasible_ind):
            # print('Both feasible')
            # % rare case of two temperatures being in the feasible range
            # % select the smaller one in an absolute sense
            T_feasible_ind = np.argmin(np.abs(Ts))
        elif np.any(T_feasible_ind):
            #One feasible
            #(Doing this so T_feasible_ind is the same for both of these cases)
            T_feasible_ind = np.argmax(T_feasible_ind)
        
        T = Ts[T_feasible_ind]
        
        if T.size == 0: #Array is empty
            # % unrealistic temperatures have been found; this is likely due
            # % to either a noise limitation or a FOV issue.
            T = 24.5
            T_solution= False
    else:
        # % Complex solutions of the temperature equation indicate that the
        # % presumed jump is too big to be accommodated by the model. The
        # % likely reason is that the jump is lost in the sensor noise.
        # % A cleaner solution would be to test the jump size versus NedL.

        # % In this case we fallback to the assumed standard temperature
        # % For the correction, no correction factors are calculated if a
        # % complex solution is found.
        T = 24.5        
        T_solution = False
    #print(T_solution)
    
    return T, Ts, T_solution


def asd_jump_correction(coeffs, spectrum, wvl, interpolate_H2O=False, iterations=3, jump_corr_method = 'hueni', negatives_corr_method='parabolic', interpolate_H2O_method='parabolic'):
    
    #negatives_corr_method = 'parabolic' or 'offset'
    #interpolate_H2O_method = 'smooth' or 'parabolic'
    #jump_corr_method      = 'hueni' or 'asdparabolic'
    
    #Change to a loop for iterations
    #Decide whether we will process one spectra at a time, or if can do multiple
    
    #Spectrum here is a single spectrum
    spectrum     = np.squeeze(spectrum) #Make sure dimension consistent
    spectrum_dim = spectrum.shape
    
    wvl     = np.squeeze(wvl) #Should be a vector 350 nm:2500nm (2151 elements)
    wvl_dim = wvl.shape
    

    #Set up jump size matrix to keep track of convergence over iterations
    jump_size_matrix = np.zeros((iterations,2)) #iteration, vnir or swir2 jump size
    
    processing_notes = []

    #VNIR Processing indices
    i725  = get_closest_wvl_index(wvl, 725)[0]
    i1000 = get_closest_wvl_index(wvl, 1000)[0] #vnir_range_end
    i1001 = get_closest_wvl_index(wvl, 1001)[0] #swir_range_start
    
    #Mid-SWIR1 splice band
    i1726 = get_closest_wvl_index(wvl, 1726)[0] #splice_band

    #SWIR2 Processing indices
    i1800 = get_closest_wvl_index(wvl, 1800)[0] #swir1_range_end
    i1801 = get_closest_wvl_index(wvl, 1801)[0] #swir2_range_start
    i1950 = get_closest_wvl_index(wvl, 1950)[0]
    
    #SWIR2 splice band
    i1960 = get_closest_wvl_index(wvl, 1960)[0]

    ### Pre-process negative radiances in VNIR ###
    # % Special handling for water and other very dark targets that feature negative radiances in the VNIR. Such problems may appear if the radiance signals are very low and the dark current correction (automatic process in the ASD) results in negative digital numbers.
    # % It would appear that this could happen if the dark current fluctuates slightly over time, or if there are some non-linearities for low radiances.
    # % The correction assumption is as follows: the dark current resulting in negative DNs in the VNIR detector is a following the parabolic function.

    negative_value_index = spectrum[:i1000+1] < 0 #Spectrum through end of the VNIR range
    percent_negative_numbers = 100*np.sum(negative_value_index)/negative_value_index.shape[0]
    
    dark_current_corrected = False

    # % Decide if a dark current issue exists.
    # % The 1 percent threshold as chosen arbitrarily and may need adapting.
    # % Typically, water spectra can have around 10% of negative radiance
    # % band values in the VNIR.
    # % Potentially, one could choose to correct every instance of negative
    # % radiance, but more research into the matter is required first.

    if percent_negative_numbers > 1 and spectrum[i1000] < 0:
        print('Correcting negative numbers')
        #Percentage of negative numbers in VNIR is >1% and the last VNIR value is negative
        # % This appears to be a case that needs dark current correcting
        # % before a jump correction can be attempted.

        # % Two methods present themselves:
        # % a) Assume a fixed offset for all bands of the VNIR channel
        # % b) Assume a parabolic offset

        # % At this point the parabolic assumption is supported by
        # % preliminary measurements. The correction method can chosen by a flag.

        if negatives_corr_method == 'offset':
            # % Try to use minimum value to correct the spectrum.
            # % Using the last band for correction is also not straightforward, as a
            # % zero value leads to infinity when trying to calculate a
            # % correction factor. Therefore, in case the last band value is the minimum, then the standard deviation of the last two bands is added as correction factor.
            min_val     = np.min(spectrum[:i1000+1])
            min_val_ind = np.argmin(spectrum[:i1000+1])

            if min_val_ind != i1000:
                # % the last band is not the smallest value, hence, no
                # % problem with the later correction
                spectrum[:i1000+1] = spectrum[:i1000+1]-min_val
            else:
                # % the last band is the smallest value. It must be greater than zero, hence, add the estimated noise of the last 2 bands to raise above zero.
                spectrum[:i1000+1] = spectrum[:i1000+1]-min_val+np.std(spectrum[i1000-1:i1000+1])
                
            processing_notes.append(f'ASD Jump Correction: Correction of VNIR channel for a presumed dark current issue (negative radiances) using {negatives_corr_method} method')
            dark_current_corrected = True
        
        elif negatives_corr_method == 'parabolic':
            # % ASD Parabolic Correction
            corr_factors = np.ones(spectrum.shape)

            y = (((wvl-725)**2) * (spectrum[i1001] - spectrum[i1000])) / (spectrum[i1000] * (1000 - 725)**2) + 1;

            corr_factors[i725:i1000+1]=y[i725:i1000+1]

            spectrum = spectrum * corr_factors
            
            processing_notes.append(f'ASD Jump Correction: Correction of VNIR channel for a presumed dark current issue (negative radiances) using {negatives_corr_method} method')
            dark_current_corrected = True
         
        else:
            print(f'Method {negatives_corr_method} not recognized. No presumed dark current correction (negative radiances) performed.')
            processing_notes.append(f'Method {negatives_corr_method} not recognized. No presumed dark current correction (negative radiances) performed.')

    ### Jump Correction ###
    if jump_corr_method == 'hueni': 
        for iteration in range(iterations):

            # After potential dark current correction above (for when there are negative values in the VNIR spectrum):
            # % use linear extrapolation to estimate the values in the last VNIR band and
            # % first SWIR2 band

            #Fit a line on the first three bands of swir1
            p = np.polyfit(np.array([1001,1002,1003]),spectrum[i1001:i1001+3],1) 
            last_vnir_estimate = np.polyval(p,1000)
            #***Numpy docs now suggest using the Polynomial class instead of polyfit

            first_swir    = spectrum[i1001]
            last_vnir     = spectrum[i1000]

            jump_size_matrix[iteration,0] = first_swir - last_vnir # VNIR jump size

            # % catch the case that the last VNIR band is zero
            if last_vnir > 0:
                corr_factor_last_vnir = last_vnir_estimate / last_vnir
            else:
                # % substitute the last band value with zero plus the noise of the
                # % last two bands to get a positive correction factor
                corr_factor_last_vnir = last_vnir_estimate / np.std(spectrum[i1000-1:i1000+1])

            #Fit a line on the last three bands of swir1
            p = np.polyfit(np.array([1798,1799,1800]), spectrum[i1800-2:i1800+1], 1);
            first_swir2_estimate = np.polyval(p, 1801);

            first_swir2 = spectrum[i1801]
            last_swir1  = spectrum[i1800]

            jump_size_matrix[iteration,1] = last_swir1 - first_swir2 # SWIR2 jump size

            corr_factor_first_swir2 = first_swir2_estimate / first_swir2

            ### Find the outside temperature given by these gains ###
            # % goal: identify an outside temperature where the correction gains
            # % established above are met
            # % ax2 + bx + c = 0
            T_vnir, Ts_vnir, T_solution_vnir = solve_quadratic_temperature(coeffs[i1000,0],
                                                   coeffs[i1000,1],
                                                   coeffs[i1000,2]-corr_factor_last_vnir) #Last VNIR
            T_swir, Ts_swir, T_solution_swir = solve_quadratic_temperature(coeffs[i1801,0],
                                                   coeffs[i1801,1],
                                                   coeffs[i1801,2]-corr_factor_first_swir2) #First SWIR2
            #Take the mean
            outside_T = np.mean(np.array([T_vnir,T_swir]))

            # % get transformation factors using these temperatures and correct the
            # % spectrum.   
            T_vector = np.zeros(wvl.shape)
            T_vector[:i1726+1] = T_vnir # Will include the i1726 (SWIR1 split band)
            T_vector[i1726+1:] = T_swir

            spec_corr_factors = coeffs[:,0]*(T_vector**2) + coeffs[:,1]*T_vector + coeffs[:,2] #Parabolic correction factor

            ### Interpolate H2O if desired ###
            if interpolate_H2O:

                spec_corr_factors_smoothed = spec_corr_factors.copy()

                if interpolate_H2O_method == 'smooth':
                    # % Approach 1: carry out massive smoothing per detector

                    #Fit 3rd degree polynomials of spec_corr_factors in VNIR, SWIR1, and SWIR2 bands
                    #And save the fits in spec_corr_factors_smoothed for each
                    #VNIR
                    c = np.polyfit(wvl[:i1000+1], spec_corr_factors[:i1000+1], 3) 
                    spec_corr_factors_smoothed[:i1000+1] = np.polyval(c, wvl[:i1000+1]) 
                    #SWIR1
                    c = np.polyfit(wvl[i1001:i1800+1], spec_corr_factors[i1001:i1800+1], 3)
                    spec_corr_factors_smoothed[i1001:i1800+1] = np.polyval(c, wvl[i1001:i1800+1])
                    #SWIR2
                    c = np.polyfit(wvl[i1801:i1960+1], spec_corr_factors[i1801:i1960+1], 3)
                    spec_corr_factors_smoothed[i1801:i1960+1] = np.polyval(c, wvl[i1801:i1960+1])

                    # % avoid overfitting: only smooth bands 1960 nm to end in initial solution
                    if iteration == 0:
                        c = np.polyfit(wvl[i1960+1:], spec_corr_factors[i1960+1:], 2) #Only using a 2nd degree polynomial here
                        spec_corr_factors_smoothed[i1960+1:] = np.polyval(c, wvl[i1960+1:])
                    else:
                        # % Approach 2: later SWIR2 part to be set to constant corr value,
                        # % similar to the ASD parabolic correction
                        # RE note: I put this as in an else statement for the 'smooth' method
                        # However, this might function separately from the smoothing method;
                        # it is unclear from the original code what the intent was
                        spec_corr_factors_smoothed[i1960+1:] = np.ones(spec_corr_factors_smoothed[i1960+1:].shape) * spec_corr_factors_smoothed[i1960]

                elif interpolate_H2O_method == 'parabolic':
                    # % Approach 3: SWIR2 uses parabolic correction
                    #SWIR2: i1801 to the end 
                    y = (((wvl-1950)**2) * (spectrum[i1800] - spectrum[i1801])) / (spectrum[i1801] * (1800 - 1950)**2) + 1
                    spec_corr_factors_smoothed[i1801:i1950] = y[i1801:i1950]
                    spec_corr_factors_smoothed[i1950:] = 1

                    # % Approach 4: VNIR parabolic solution
                    #VNIR: start through i1000
                    y = (((wvl-725))**2 * (spectrum[i1001] - spectrum[i1000])) / (spectrum[i1000] * (1000 - 725)**2) + 1
                    spec_corr_factors_smoothed[i725+1:i1000+1] = y[i725+1:i1000+1]
                    spec_corr_factors_smoothed[:i725+1] = 1

                    # % Approach 5: SWIR1 set to constant
                    #SWIR1: i1001 through i1800
                    spec_corr_factors_smoothed[i1001:i1800+1] = 1

                spec_corr_factors = spec_corr_factors_smoothed.copy() #Copy the smoothed factors back over
        
            #If there was no valid temperature solution for a band,
            #that band will not be corrected (the correction factors will be set to one)
            # % correction for model limits due to noise
            if not T_solution_vnir:
                spec_corr_factors[:i1726+1] = 1 
                print('Attention: no VNIR correction due to noise or model limit');
                processing_notes.append('ASD Jump Correction: No VNIR correction due to noise or model limit')

            if not T_solution_swir:
                spec_corr_factors[i1726+1:] = 1
                print('Attention: no SWIR correction due to noise or model limit');
                processing_notes.append('ASD Jump Correction: No SWIR correction due to noise or model limit')
                
            #Correct the spectrum by multiplying by the correction factors
            spectrum = spectrum * spec_corr_factors

            # % recalculate the temperatures based on iterated correction
            # % coefficients     
            T_vnir, Ts_vnir, T_solution_vnir = solve_quadratic_temperature(coeffs[i1000,0],
                                               coeffs[i1000,1],
                                               coeffs[i1000,2]-spec_corr_factors[i1000])
            T_swir, Ts_swir, T_solution_swir = solve_quadratic_temperature(coeffs[i1801,0],
                                               coeffs[i1801,1],
                                               coeffs[i1801,2]-spec_corr_factors[i1801])
            outside_T = np.mean(np.array([T_vnir,T_swir]))
        
    elif jump_corr_method == 'asdparabolic':
        spec_corr_factors = np.ones(wvl.shape)
        spec_corr_factors[i725:i1000+1] = 1+((wvl[i725:i1000+1]-725)**2 * (spectrum[i1001]-spectrum[i1000]))/(spectrum[i1000]*(1000-725)**2)
        spec_corr_factors[i1801:i1950+1] = 1+((wvl[i1801:i1950+1]-1950)**2 * (spectrum[i1800]-spectrum[i1801]))/(spectrum[i1801]*(1801-1950)**2)
        spectrum = spectrum * spec_corr_factors

        jump_size_matrix[0,0] = spectrum[i1001]-spectrum[i1000]
        jump_size_matrix[0,1] = spectrum[i1800]-spectrum[i1801]

        T_vnir, Ts_vnir, T_solution_vnir = solve_quadratic_temperature(coeffs[i1000,0],
                                           coeffs[i1000,1],
                                           coeffs[i1000,2]-spec_corr_factors[i1000])
        T_swir, Ts_swir, T_solution_swir = solve_quadratic_temperature(coeffs[i1801,0],
                                           coeffs[i1801,1],
                                           coeffs[i1801,2]-spec_corr_factors[i1801])
        outside_T = np.mean(np.array([T_vnir,T_swir]))
    else:
        print(f'Method {jump_corr_method} not recognized. No jump correction performed.')
        processing_notes.append(f'Method {jump_corr_method} not recognized. No jump correction performed.')
            

    return spectrum, outside_T, spec_corr_factors, jump_size_matrix, processing_notes

    
