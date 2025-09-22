# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:53:54 2025

@author: Nikita Otstavnov
HSE University
"""


# %% LIBRARY IMPORT ###########################################################

import os 
import mne
import pandas as pd
import numpy as np
import mne
# %matplotlib qt
import matplotlib.pyplot as plt
import mne_connectivity
import math
import pingouin as pg
import seaborn as sns
import matplotlib
from scipy.integrate import simps #!!! simpson
from matplotlib.image import imread
import matplotlib.gridspec as gridspec
import time
import matplotlib.image as mpimg
        

# %% EPOCHING

def epochs_creation(Data_path, subject, final_name, annotation_final_name, 
                    subject_ductionary, tmin_input, tmax_input, 
                    movie_duration_input, window_for_visualization,
                    scalings_for_visualization):    

    #### DOWNLOADER ###############################################################
    if int(subject) not in (18, 20):
        os.chdir('{}'.format(Data_path))
        raw_new                              = mne.io.read_raw_edf('{}{}.edf'.format(subject, final_name))
        annot_from_file                      = mne.read_annotations("{}{}.csv".format(subject, annotation_final_name))
      
        ###############################################################################
        def prepare_data(subject):
            
            # EDF Download
            file                             = Data_path + '/' + subject + '_upd2.edf'
            folder                           = Data_path + '/' 
            
            # Psychopy file mapping to the triggers
            stimulus_f                       = Data_path + '/'  +  subject + '_psychopy.csv'
            stimulus_file                    = pd.read_csv(stimulus_f)
            len(stimulus_file)
            len(stimulus_file[:-1])
            stimulus_file                    = stimulus_file[:-1]
        
            # Mapping
            from Peak_finder_functions import find_events_eeg, label_define
            table_with_time, raw_data        = find_events_eeg(file)
            events                           = label_define(stimulus_file, table_with_time)
            raw_data.set_annotations(annot_from_file)
            raw_data.load_data()
            raw_new                          = raw_data
                
            return raw_new, events
     
        # Subjects with changed data (special cases)
        raw_new, events = prepare_data(subject)
        raw_new.plot(events=events, scalings=scalings_for_visualization , duration=window_for_visualization)
             
        ###########################################################################
        def subj_analyzer(subject):
            what_dict_use = subject_ductionary['sub{}'.format(subject)]
            
            ch_names1 = what_dict_use['Left Amygdala']
            ch_names2 = what_dict_use['Left Orbitofrontal']
            ch_names3 = what_dict_use['Left Hippocampus']
            ch_names4 = what_dict_use['Left Insula']
            ch_names5 = what_dict_use['Right Amygdala']
            ch_names6 = what_dict_use['Right Orbitofrontal']
            ch_names7 = what_dict_use['Right Hippocampus']
            ch_names8 = what_dict_use['Right Insula']
        
            total_ch_names =  ch_names1 + ch_names2 + ch_names3 + ch_names4 + ch_names5 + ch_names6 + ch_names7 + ch_names8
            return total_ch_names
        
        ###########################################################################
        
        # Raw file merger with trigges
        total_ch_names = subj_analyzer(subject)
        raw_new.pick_channels(total_ch_names)
        
    if int(subject) in (18, 20):
        os.chdir('{}'.format(Data_path))
        print("Special case")
        raw_new              = mne.io.read_raw_edf('{}{}.edf'.format(subject, final_name))
        events               = mne.read_events('{}-events.eve'.format(subject))
        raw_new.plot(events=events, scalings=scalings_for_visualization , duration=window_for_visualization)
        
    ##  EPOCHING long format ##################################################
   
    events_2              = events.astype(int)
  
    def create_180_epochs_for_condition(raw, events, event_marker, 
                                        tmin=tmin_input, tmax=tmax_input, 
                                        movie_duration=movie_duration_input):
       
        """
        Create a single MNE Epochs object containing 180 2-second epochs for a specified condition.
        
        Parameters:
        - raw: MNE Raw object containing the EEG data
        - events: Numpy array of events (shape: [n_events, 3])
        - event_marker: Integer, the event marker for the condition to analyze
        - tmin: Start time for each epoch (default = 0s)
        - tmax: End time for each epoch (default = 2s)
        - movie_duration: Duration of each movie in seconds (default = 60s)
        
        Returns:
        - MNE Epochs object containing 180 2-second epochs for the specified condition.
        """
        # Filter the events to only include those with the specified event_marker
        condition_events = events[events[:, 2] == event_marker]
    
        # Initialize a list to store all new event samples for 2-second epochs
        new_events = []
        
        sfreq = raw.info['sfreq']  # Sampling frequency to convert time to samples
        
        # For each event in the condition
        for event in condition_events:
            start_sample = event[0]  # The sample corresponding to the event
    
            # For each movie, split it into 30 epochs (2 seconds each)
            for i in range(0, movie_duration, tmax - tmin):
                epoch_start_sample = int(start_sample + i * sfreq)  # Convert time to samples
                new_event = [epoch_start_sample, 0, event_marker]
                new_events.append(new_event)
        
        # Convert the list of new events into a numpy array
        new_events = np.array(new_events)
        len(new_events)
        
        # Create the MNE Epochs object for all 180 events (2 seconds each)
        epochs = mne.Epochs(raw, new_events, event_id= event_marker, 
                            tmin=tmin, tmax=tmax, baseline=None, preload=True)
        
        return epochs
    
    event_marker = 22 
    epochs_s = create_180_epochs_for_condition(raw_new, events_2, event_marker)
    len(epochs_s)
    
    event_marker = 32
    epochs_r = create_180_epochs_for_condition(raw_new, events_2, event_marker)
    len(epochs_r)
    
    event_marker = 12
    epochs_wp = create_180_epochs_for_condition(raw_new, events_2, event_marker)
    len(epochs_wp)
    
    event_marker = 72
    epochs_wn = create_180_epochs_for_condition(raw_new, events_2, event_marker)
    len(epochs_wn)
    
    epochs_list_long_events = [epochs_wp, epochs_s, epochs_r, epochs_wn]
    
    return epochs_list_long_events, raw_new


# %% POWER SPECTRAL DENSITY

def psd_calculations(n_jobs, fmin, fmax, method, epochs_list_long_events, 
                     verbose_input, proj_input, tmin_input, tmax_input, average_input):
  
    """
    CHECK LABELS, if something you will need to recreate them 
    """
    
    def psd_calculation(n_jobs, fmin, fmax, method, epochs_list_long_events):
        psd_wp_list = []
        i = 0
        epochs_watchpos = epochs_list_long_events[0]
        for i in range(len(epochs_watchpos)):
            epochs = epochs_watchpos[i]
            psd_wp = epochs.compute_psd(method=method, 
                                        fmin=fmin, fmax=fmax, 
                                        tmin=tmin_input, tmax=tmax_input, 
                                        # picks=ch_inc, 
                                        proj=proj_input , 
                                        n_jobs=n_jobs, 
                                        verbose=verbose_input)
            psd_wp_list.append(psd_wp)
            

        psd_s_list = []
        i = 0
        epochs_suppress = epochs_list_long_events[1]
        for i in range(len(epochs_suppress)):
            epochs = epochs_suppress[i]
            psd_s = epochs.compute_psd(method=method, 
                                        fmin=fmin, fmax=fmax, 
                                        tmin=tmin_input, tmax=tmax_input, 
                                        # picks=ch_inc, 
                                        proj=proj_input, 
                                        n_jobs=n_jobs, 
                                        verbose=verbose_input)
            psd_s_list.append(psd_s)
        
        
        psd_r_list = []
        i = 0
        epochs_reassess = epochs_list_long_events[2]
        for i in range(len(epochs_reassess)):
            epochs = epochs_reassess[i]
            if len(epochs.events) == 0:
                continue
            else: 
                psd_r = epochs.compute_psd(method=method, 
                                            fmin=fmin, fmax=fmax, 
                                            tmin=tmin_input, tmax=tmax_input, 
                                            # picks=ch_inc, 
                                            proj=proj_input, 
                                            n_jobs=n_jobs, 
                                            verbose=verbose_input)
                psd_r_list.append(psd_r)
        
        psd_wn_list = []
        i = 0
        epochs_watchneu = epochs_list_long_events[3]
        for i in range(len(epochs_watchneu)):
            epochs = epochs_watchneu[i]
            if len(epochs.events) == 0:
                continue
            else: 
                psd_wn = epochs.compute_psd(method=method, 
                                            fmin=fmin, fmax=fmax, 
                                            tmin=tmin_input, tmax=tmax_input, 
                                            # picks=ch_inc, 
                                            proj=proj_input, 
                                            n_jobs=n_jobs, 
                                            verbose=verbose_input)
                psd_wn_list.append(psd_wn)
            
        return psd_wp_list, psd_s_list, psd_r_list, psd_wn_list, epochs_watchpos, epochs_suppress, epochs_reassess, epochs_watchneu
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    psd_wp_list, psd_s_list, psd_r_list, psd_wn_list, epochs_watchpos, epochs_suppress, epochs_reassess, epochs_watchneu = psd_calculation(n_jobs, fmin, fmax, method, epochs_list_long_events)
    psd_condition_average = psd_wn_list + psd_wp_list + psd_s_list + psd_r_list
    
    len(psd_wp_list)
    len(psd_s_list)
    len(psd_r_list)
    len(psd_wn_list)
    len(psd_condition_average)
    
    freqs = psd_wp_list[0].freqs
    ###############################################################################
    ####### AVERAGE ACROSS EPOCHS #################################################
    
    def psd_average_across_epochs(psd_wp_list, psd_s_list, psd_r_list, psd_wn_list, psd_condition_average):
        
        i = 0 
        psd_epochs_wn = []
        for i in range(len(psd_wn_list)):
            psd_to_work_with = psd_wn_list[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_wn.append(psd_data )
            
        np.shape(psd_epochs_wn)
        psd_epochs_wn = np.array(psd_epochs_wn)
        psd_wn        = np.average(psd_epochs_wn, axis = 0)
        np.shape(psd_wn)

         
        i = 0 
        psd_epochs_wp = []
        for i in range(len(psd_wp_list)):
            psd_to_work_with = psd_wp_list[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_wp.append(psd_data )
            
        np.shape(psd_epochs_wp)
        psd_epochs_wp = np.array(psd_epochs_wp)
        psd_wp        = np.average(psd_epochs_wp, axis = 0)
        np.shape(psd_wp)
        
        i = 0 
        psd_epochs_s = []
        for i in range(len(psd_s_list)):
            psd_to_work_with = psd_s_list[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_s.append(psd_data )
            
        np.shape(psd_epochs_s)
        psd_epochs_s = np.array(psd_epochs_s)
        psd_s        = np.average(psd_epochs_s, axis = 0)
        np.shape(psd_s)
        
        i = 0 
        psd_epochs_r = []
        for i in range(len(psd_r_list)):
            psd_to_work_with = psd_r_list[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_r.append(psd_data )
            
        np.shape(psd_epochs_r)
        psd_epochs_r = np.array(psd_epochs_r)
        psd_r        = np.average(psd_epochs_r, axis = 0)
        np.shape(psd_r)
        
        i = 0 
        psd_epochs_ca = []
        for i in range(len(psd_condition_average)):
            psd_to_work_with = psd_condition_average[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_ca.append(psd_data )
            
        np.shape(psd_epochs_ca)
        psd_epochs_ca = np.array(psd_epochs_ca)
        psd_CA        = np.average(psd_epochs_ca, axis = 0)
        np.shape(psd_CA)
        
        return psd_wn, psd_wp, psd_s, psd_r, psd_CA
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    psd_wn, psd_wp, psd_s, psd_r,psd_CA = psd_average_across_epochs(psd_wp_list, 
                                                                    psd_s_list, 
                                                                    psd_r_list, 
                                                                    psd_wn_list,
                                                                    psd_condition_average)
    np.shape(psd_wn)
    np.shape(psd_CA)
    
    ############# OBJECT DATA TRANSFORMER
    psd_r_list[0]._data  = psd_r
    psd_s_list[0]._data  = psd_s
    psd_wp_list[0]._data = psd_wp
    psd_wn_list[0]._data = psd_wn
    psd_wn_list[0]._data = psd_wn
    
    wp_psd = psd_wp_list[0]
    wn_psd = psd_wn_list[0]
    r_psd  = psd_r_list[0] 
    s_psd  = psd_s_list[0] 
    
    wp_psd.plot(average=average_input)
    wn_psd.plot(average=average_input)
    r_psd.plot(average=average_input)
    s_psd.plot(average=average_input)

    return wp_psd, wn_psd, r_psd, s_psd, epochs_suppress


def psd_visualizer(fmin, fmax, ch, wp_psd, wn_psd, r_psd, s_psd): 

    matplotlib.rc('ytick', labelsize=25)
    matplotlib.rc('xtick', labelsize=25)
    
    psd_r  = r_psd.copy().pick(ch)
    psd_s  = s_psd.copy().pick(ch)
    psd_wn = wn_psd.copy().pick(ch)
    psd_wp = wp_psd.copy().pick(ch)
    
    psd_r_data  = psd_r.get_data()
    psd_s_data  = psd_s.get_data()
    psd_wn_data = psd_wn.get_data()
    psd_wp_data = psd_wp.get_data()
    
    psd_r_data_log = np.log10(psd_r_data) 
    psd_s_data_log = np.log10(psd_s_data) 
    psd_wn_data_log = np.log10(psd_wn_data) 
    psd_wp_data_log = np.log10(psd_wp_data) 
    
    ymin_lg = np.min([psd_r_data_log,psd_s_data_log, psd_wn_data_log,  psd_wp_data_log ]) - 0.1
    ymax_lg = np.max([[psd_r_data_log,psd_s_data_log, psd_wn_data_log,  psd_wp_data_log ]]) + 0.1 
    ymin_lg = -13.5
    ymax_lg = -9

    fig, ax = plt.subplots(1,1, layout='constrained')
    ax.plot(range(fmax), psd_s_data_log[0][0,fmin-2:fmax], color = 'blue', label = 'Suppress')
    ax.plot(range(fmax), psd_r_data_log[0][0,fmin-2:fmax], color = 'green', label = 'Reappraise')
    ax.plot(range(fmax), psd_wp_data_log[0][0,fmin-2:fmax], color = 'orange', label = 'Watch positive')
    ax.plot(range(fmax), psd_wn_data_log[0][0,fmin-2:fmax], color = 'black', label = 'Watch neutral')
    ax.set_ylim([ymin_lg, ymax_lg])
    ax.set_xlim(left=0, right=fmax-1) 
    ax.legend(fontsize=20)
    plt.ylabel('Power spectrum density [log(dB)]', fontweight='bold', fontsize = 20)
    plt.xticks(np.arange(0, fmax, step=2), fontsize=25)
    plt.xlabel('Frequency (Hz)', fontweight='bold', fontsize = 20)
    ax.set_title('{}'.format(ch))
    
    return fig, ax


# %% PHASE LOCKING VALUE

def plv_calculator(subject, subject_ductionary, raw_new, fmin, fmax, n_jobs, what_to_take_from_the_dict, 
                   epochs_s, epochs_r, epochs_wp, epochs_wn, epochs_suppress, method_input,
                   faverage_input, verbose_input, fskip_input): 

    ###########################################################################
    def define_ch_pairs(epochs_suppress, subject, what_to_take_from_the_dict):
        #### CHANNEL SELECTION
        example_epoch = epochs_suppress[0]  
        dict_name = 'sub' + subject
        dictionary_we_work_with_1 = subject_ductionary[dict_name]
      
        d = {}
         
        # iterating through the elements of list
        for i in what_to_take_from_the_dict:
            d[i] = dictionary_we_work_with_1[i]
            
        dictionary_we_work_with = d
        
        # all_channels    = example_epoch.ch_names
        # len(all_channels)
        all_channels = [value for key in what_to_take_from_the_dict for value in
                        dictionary_we_work_with.get(key, [])]
        len(all_channels)
        
        # List of channel names
        # Library with key as channel name and info as index
        channel_to_index = {name: idx for idx, name in enumerate(all_channels)}
        # The same operation but in reverse order
        index_to_channel = {idx: name for idx, name in enumerate(all_channels)}
    
        # Brain areas list
        indices_pairs = []
        
        # Unique pair of indices
        unique_indices_pairs = set()
        # List of unique pairs of channels
        unique_names_pairs = set()
           
        # Check the belonging to either left or right hemisphere
        def is_left_zone(zone_name):
            return 'Left' in zone_name
        
        def is_right_zone(zone_name):
            return 'Right' in zone_name
        
        for zone_1, channels_1 in dictionary_we_work_with.items():
            for zone_2, channels_2 in dictionary_we_work_with.items():
                # We check that the zone belongs to one hemisphere and that these are different zones, but with the mode
                if ((is_left_zone(zone_1) and is_left_zone(zone_2)) or 
                    (is_right_zone(zone_1) and is_right_zone(zone_2))) and channels_1 and channels_2:
                    
                    # Skip if it's the same zone
                    if zone_1 == zone_2:
                        continue  # Skip the iteration if the channels are from the same zone
                       
                    # We cycle through channels from the first and second zones
                    for ch1 in channels_1:
                        for ch2 in channels_2:
                            # We get channel indexes by their names
                            if ch1 in all_channels and ch2 in all_channels:
                                ch1_idx = all_channels.index(ch1)
                                ch2_idx = all_channels.index(ch2)
                                
                                # Add a couple of indexes if these are different channels
                                if ch1_idx != ch2_idx:
                                    pair = tuple(sorted((ch1_idx, ch2_idx)))  # Sort to always get (min, max)
                                    unique_indices_pairs.add(pair)
                                
                                    # Add the corresponding channel names
                                    name_pair = tuple(sorted((ch1, ch2)))  # We also sort for unification
                                    unique_names_pairs.add(name_pair)
                                    
        # Convert the index pairs into the format the function expects
        indices = ([pair[0] for pair in unique_indices_pairs], [pair[1] for pair in unique_indices_pairs])
        
        # Converting channel name pairs
        indices_names = ([pair[0] for pair in unique_names_pairs], [pair[1] for pair in unique_names_pairs])
    
        return unique_indices_pairs, indices,indices_names,  index_to_channel
    
    
    ###############################################################################
    ########################################## SELECT PARTICULAR CHANNELS #########
    ###############################################################################
    
    unique_indices_pairs, indices, indices_names, index_to_channel = define_ch_pairs(epochs_s, subject, what_to_take_from_the_dict)
 
    
    ###############################################################################
    ########################################## PLV CALCULATION ####################
    ###############################################################################
    i = 0 
    suppress_array = []
    reassess_array = []
    watchpos_array = []
    watchneu_array = []
        
    datas   = epochs_s #array_like, shape=(n_epochs, n_signals, n_times) 
    datar   = epochs_r 
    datawp  = epochs_wp 
    datawn  = epochs_wn 
    raw1 = raw_new
    
    #### PLV CALCULATION

    outcome_wn = mne_connectivity.spectral_connectivity_epochs(datawn, 
                                                                names=None, 
                                                                method=method_input, 
                                                                indices=indices, 
                                                                sfreq=raw1.info['sfreq'], 
                                                                fmin=fmin, fmax=fmax, 
                                                                fskip=fskip_input, 
                                                                faverage=faverage_input, 
                                                                n_jobs=n_jobs, 
                                                                verbose=verbose_input)  
    
    
    outcome_wp = mne_connectivity.spectral_connectivity_epochs(datawp, 
                                                                names=None, 
                                                                method=method_input, 
                                                                indices=indices, 
                                                                sfreq=raw1.info['sfreq'], 
                                                                fmin=fmin, fmax=fmax, 
                                                                fskip=fskip_input, 
                                                                faverage=faverage_input, 
                                                                n_jobs=n_jobs, 
                                                                verbose=verbose_input)  
    
    outcome_r = mne_connectivity.spectral_connectivity_epochs(datar, 
                                                              names=None, 
                                                              method=method_input, 
                                                              indices=indices, 
                                                              sfreq=raw1.info['sfreq'], 
                                                              fmin=fmin, fmax=fmax, 
                                                              fskip=fskip_input, 
                                                              faverage=faverage_input, 
                                                              n_jobs=n_jobs, 
                                                              verbose=verbose_input)  
     
     
    outcome_s = mne_connectivity.spectral_connectivity_epochs(datas, 
                                                              names=None, 
                                                              method=method_input, 
                                                              indices=indices, 
                                                              sfreq=raw1.info['sfreq'], 
                                                              fmin=fmin, fmax=fmax, 
                                                              fskip=fskip_input, 
                                                              faverage=faverage_input, 
                                                              n_jobs=n_jobs, 
                                                              verbose=verbose_input)  
   
    #### NUMPER
    suppress_array_np = np.array(outcome_s.get_data())
    reassess_array_np = np.array(outcome_r.get_data())
    watchpos_array_np = np.array(outcome_wp.get_data())
    watchneu_array_np = np.array(outcome_wn.get_data())

    np.shape(suppress_array_np)
    mean_supr = suppress_array_np
    mean_reas = reassess_array_np
    mean_wpos = watchpos_array_np
    mean_wneu = watchneu_array_np
    np.shape(mean_wneu)
    
    return outcome_wn, outcome_wp, outcome_r, outcome_s, index_to_channel, mean_wneu, mean_wpos, mean_reas, mean_supr


def calculate_auc_at_selected_frequencies(frequencies, plv_watch_neutral, 
                                          plv_watch_positive, plv_suppress, 
                                          plv_reassess, selection_range, 
                                          j, lw, fs, fmin_to_plot):
    fig, ax = plt.subplots(1,1)
   
    ax.plot(frequencies, plv_suppress, label="Suppress", color = 'blue', linewidth=lw)
    ax.plot(frequencies, plv_reassess, label="Reappraise", color = 'green', linewidth=lw) #, alpha = 0.5
    ax.plot(frequencies,plv_watch_neutral, label="Watch Neutral", color = 'black', linewidth=lw) #, alpha = 0.5
    ax.plot(frequencies, plv_watch_positive, label="Watch Positive", color = 'orange', linewidth=lw) #, alpha = 0.5
   
    ax.axhline(0, color='black', linestyle='--', linewidth=lw-1)
    
    ###### MIN-MAXER
    plv_arrays = [
        plv_suppress,
        plv_reassess,
        plv_watch_neutral,
        plv_watch_positive
    ]
    # plv_min = np.min([arr.min() for arr in plv_arrays])
    plv_max = np.max([arr.max() for arr in plv_arrays])
   
    
    ###### LABELS
    y_min = 0
    y_max = plv_max + 0.05
    y_step = 0.1
    
  
    ax.set_yticks(np.arange(y_min, y_max, step=y_step+0.02))
    plt.yticks(fontsize=fs)
   
    fmin = fmin_to_plot
    fmax = frequencies[-1] 
    plt.xticks(np.arange(fmin+3, fmax, step=5), fontsize=fs)
    
    ax.set_xlim(left=fmin, right=fmax-1) 

    def scale_ticks(value, pos):
        
        return f"{value * 0.1:.2f}"

    ax.legend()
    ax.margins(x=0) 
    
    # Add title, labels, and legend
    plt.xlabel('Frequency (Hz)', fontsize=fs+2, fontweight='bold')
    plt.ylabel('Phase Locking Value', fontsize=fs+2, labelpad=20, fontweight='bold')
    plt.legend(fontsize=fs+2)
    
    # Allow the user to select points on the plot
    plt.show()
    print("Please click on the plot to select points where you want to calculate the AUC.")
    selected_points = plt.ginput(n=-1, timeout=0)  # Allow the user to select as many points as desired
    
    # Process selected points
    selected_frequencies = [point[0] for point in selected_points]
    print(f"Selected frequencies: {selected_frequencies}")
    
    # Store the AUC results
    auc_results = []
    
    for selected_freq in selected_frequencies:
        # Determine the frequency range (selected frequency ± selection_range Hz)
        lower_bound = selected_freq - selection_range
        upper_bound = selected_freq + selection_range
        
        # Find the indices corresponding to the frequency range
        indices_in_range = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]
        
        # Calculate the area under the curve for each condition
        auc_watch_neutral = simps(plv_watch_neutral[indices_in_range], frequencies[indices_in_range])
        auc_watch_positive = simps(plv_watch_positive[indices_in_range], frequencies[indices_in_range])
        auc_suppress = simps(plv_suppress[indices_in_range], frequencies[indices_in_range])
        auc_reassess = simps(plv_reassess[indices_in_range], frequencies[indices_in_range])
        
        list_y_axis = [plv_watch_neutral[indices_in_range], plv_watch_positive[indices_in_range], plv_suppress[indices_in_range], plv_reassess[indices_in_range]]
        max_plv = np.max(list_y_axis)
        
        ax.axvline(lower_bound, color='black', linestyle='--', linewidth=lw-1)  
        ax.axvline(upper_bound, color='black', linestyle='--', linewidth=lw-1)
        
        # Append the result for this selected frequency
        auc_results.append({
            'selected_frequency': selected_freq,
            'watch_neutral_auc': auc_watch_neutral,
            'watch_positive_auc': auc_watch_positive,
            'suppress_auc': auc_suppress,
            'reassess_auc': auc_reassess
        })
        
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(1)  # 1-second delay
    plt.savefig('PLV.png', dpi=300, bbox_inches='tight')  # Recommended
    fig.show()
    
    return auc_results, fig,  fmin, fmax 

# %% FIGURE 4 PREPS
    
def psd_for_combined_figure(ch_name, method_psd_input, fmin, fmax, epochs_list_long_events, verbose_input,
                                            tmin_input, tmax_input, proj_input, lw, n_jobs, fs, result,
                                            selection_range): 
    
    extracted_part1 = ch_name.split(' - ')[0]
    extracted_part2 = ch_name.split(' - ')[1]
    
    ch_inc = [extracted_part1, extracted_part2]
    print(ch_inc)
    method = method_psd_input
    ###############################################################################
    #### CHECK LABELS, if something you will need to recreate them
    ###############################################################################
    def psd_calculation(n_jobs, fmin, fmax, method, epochs_list_long_events):
        psd_wp_list = []
        i = 0
        epochs_watchpos = epochs_list_long_events[0]
        for i in range(len(epochs_watchpos)):
            epochs = epochs_watchpos[i]
            psd_wp = epochs.compute_psd(method=method, 
                                        fmin=fmin, fmax=fmax, 
                                        tmin=tmin_input, tmax=tmax_input, 
                                        picks=ch_inc, 
                                        proj=proj_input, 
                                        n_jobs=n_jobs, 
                                        verbose=verbose_input)
            psd_wp_list.append(psd_wp)
            
        
        psd_s_list = []
        i = 0
        epochs_suppress = epochs_list_long_events[1]
        for i in range(len(epochs_suppress)):
            epochs = epochs_suppress[i]
            psd_s = epochs.compute_psd(method=method, 
                                       fmin=fmin, fmax=fmax, 
                                       tmin=tmin_input, tmax=tmax_input, 
                                       picks=ch_inc, 
                                       proj=proj_input, 
                                       n_jobs=n_jobs, 
                                       verbose=verbose_input)
            psd_s_list.append(psd_s)
        
        
        psd_r_list = []
        i = 0
        epochs_reassess = epochs_list_long_events[2]
        for i in range(len(epochs_reassess)):
            epochs = epochs_reassess[i]
            if len(epochs.events) == 0:
                continue
            else: 
                psd_r = epochs.compute_psd(method=method, 
                                            fmin=fmin, fmax=fmax, 
                                            tmin=tmin_input, tmax=tmax_input, 
                                            picks=ch_inc, 
                                            proj=proj_input, 
                                            n_jobs=n_jobs, 
                                            verbose=verbose_input)
                psd_r_list.append(psd_r)
        
        psd_wn_list = []
        i = 0
        epochs_watchneu = epochs_list_long_events[3]
        for i in range(len(epochs_watchneu)):
            epochs = epochs_watchneu[i]
            if len(epochs.events) == 0:
                continue
            else: 
                psd_wn = epochs.compute_psd(method=method, 
                                            fmin=fmin, fmax=fmax, 
                                            tmin=tmin_input, tmax=tmax_input, 
                                            picks=ch_inc, 
                                            proj=proj_input, 
                                            n_jobs=n_jobs, 
                                            verbose=verbose_input)
                psd_wn_list.append(psd_wn)
            
        return psd_wp_list, psd_s_list, psd_r_list, psd_wn_list
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    psd_wp_list, psd_s_list, psd_r_list, psd_wn_list = psd_calculation(n_jobs, fmin, fmax, method, epochs_list_long_events)
    psd_condition_average = psd_wn_list + psd_wp_list + psd_s_list + psd_r_list
    
    len(psd_wp_list)
    len(psd_s_list)
    len(psd_r_list)
    len(psd_wn_list)
    len(psd_condition_average)
    psd_s_list[0].info['ch_names']
    
    freqs = psd_wp_list[0].freqs
    ###############################################################################
    ####### AVERAGE ACROSS EPOCHS #################################################
    
    def psd_average_across_epochs(psd_wp_list, psd_s_list, psd_r_list, psd_wn_list, psd_condition_average):
        
        i = 0 
        psd_epochs_wn = []
        for i in range(len(psd_wn_list)):
            psd_to_work_with = psd_wn_list[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_wn.append(psd_data )
            
        np.shape(psd_epochs_wn)
        psd_epochs_wn = np.array(psd_epochs_wn)
        psd_wn        = np.average(psd_epochs_wn, axis = 0)
        # psd_wn        = np.average(psd_wn, axis = 0)
        np.shape(psd_wn        )
    
         
        i = 0 
        psd_epochs_wp = []
        for i in range(len(psd_wp_list)):
            psd_to_work_with = psd_wp_list[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_wp.append(psd_data )
            
        np.shape(psd_epochs_wp)
        psd_epochs_wp = np.array(psd_epochs_wp)
        psd_wp        = np.average(psd_epochs_wp, axis = 0)
        # psd_wp        = np.average(psd_wp, axis = 0)
        np.shape(psd_wp        )
        
        i = 0 
        psd_epochs_s = []
        for i in range(len(psd_s_list)):
            psd_to_work_with = psd_s_list[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_s.append(psd_data )
            
        np.shape(psd_epochs_s)
        psd_epochs_s = np.array(psd_epochs_s)
        psd_s        = np.average(psd_epochs_s, axis = 0)
        # psd_s        = np.average(psd_s, axis = 0)
        np.shape(psd_s        )
        
        i = 0 
        psd_epochs_r = []
        for i in range(len(psd_r_list)):
            psd_to_work_with = psd_r_list[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_r.append(psd_data )
            
        np.shape(psd_epochs_r)
        psd_epochs_r = np.array(psd_epochs_r)
        psd_r        = np.average(psd_epochs_r, axis = 0)
        # psd_r        = np.average(psd_r, axis = 0)
        np.shape(psd_r        )
        
        i = 0 
        psd_epochs_ca = []
        for i in range(len(psd_condition_average)):
            psd_to_work_with = psd_condition_average[i]
            psd_data = psd_to_work_with.get_data()
            psd_epochs_ca.append(psd_data )
            
        np.shape(psd_epochs_ca)
        psd_epochs_ca = np.array(psd_epochs_ca)
        psd_CA        = np.average(psd_epochs_ca, axis = 0)
        # psd_CA        = np.average(psd_CA, axis = 0)
        np.shape(psd_CA        )
        
        
        return psd_wn, psd_wp, psd_s, psd_r, psd_CA
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    psd_wn, psd_wp, psd_s, psd_r,psd_CA = psd_average_across_epochs(psd_wp_list, 
                                                                    psd_s_list, 
                                                                    psd_r_list, 
                                                                    psd_wn_list,
                                                                    psd_condition_average)
    np.shape(psd_wn)
    np.shape(psd_CA)
    # psd_wn.info['ch_names']
    
    ############# OBJECT DATA TRANSFORMER
    psd_r_list[0]._data  = psd_r
    psd_s_list[0]._data  = psd_s
    psd_wp_list[0]._data = psd_wp
    psd_wn_list[0]._data = psd_wn
    # psd_ca_list[0]._data = psd_CA
    
    wp_psd = psd_wp_list[0]
    wn_psd = psd_wn_list[0]
    r_psd  = psd_r_list[0] 
    s_psd  = psd_s_list[0] 
    
    
    #### VISUALIZATION
    freqs = r_psd._freqs
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    # CHANNEL 1 VALUES
    extracted_part = ch_name.split(' - ')[0]
    print(extracted_part)
    ch = extracted_part

    psd_r  = r_psd.copy().pick(ch)
    psd_s  = s_psd.copy().pick(ch)
    psd_wn = wn_psd.copy().pick(ch)
    psd_wp = wp_psd.copy().pick(ch)

    psd_r_data  = psd_r.get_data()
    psd_s_data  = psd_s.get_data()
    psd_wn_data = psd_wn.get_data()
    psd_wp_data = psd_wp.get_data()

    psd_r_data_log = np.log10(psd_r_data[0,0,freq_mask]) 
    psd_s_data_log = np.log10(psd_s_data[0,0,freq_mask]) 
    psd_wn_data_log = np.log10(psd_wn_data[0,0,freq_mask]) 
    psd_wp_data_log = np.log10(psd_wp_data[0,0,freq_mask]) 


    # CHANNEL 2 VALUES
    extracted_part = ch_name.split(' - ')[1]
    print(extracted_part)
    ch = extracted_part

    psd_r2  = r_psd.copy().pick(ch)
    psd_s2  = s_psd.copy().pick(ch)
    psd_wn2 = wn_psd.copy().pick(ch)
    psd_wp2 = wp_psd.copy().pick(ch)

    psd_r_data2  = psd_r2.get_data()
    psd_s_data2  = psd_s2.get_data()
    psd_wn_data2 = psd_wn2.get_data()
    psd_wp_data2 = psd_wp2.get_data()

    psd_r_data_log2 = np.log10(psd_r_data2[0,0,freq_mask]) 
    psd_s_data_log2 = np.log10(psd_s_data2[0,0,freq_mask]) 
    psd_wn_data_log2 = np.log10(psd_wn_data2[0,0,freq_mask]) 
    psd_wp_data_log2 = np.log10(psd_wp_data2[0,0,freq_mask]) 

    ymin_lg = np.min([psd_r_data_log2,psd_s_data_log2, psd_wn_data_log2,  psd_wp_data_log2, 
                      psd_r_data_log, psd_s_data_log,  psd_wn_data_log,   psd_wp_data_log  ]) 
    ymax_lg = np.max([psd_r_data_log2,psd_s_data_log2, psd_wn_data_log2,  psd_wp_data_log2, 
                      psd_r_data_log, psd_s_data_log,  psd_wn_data_log,   psd_wp_data_log  ])

    ymin_lg = round(ymin_lg)
    ymax_lg = round(ymax_lg)


    ###### CHANNEL 1 PLOT
    fig_psd1, ax = plt.subplots(1,1) #Creating a plot

    ## Fulfilling the plot
    ax.plot(range(fmin, round(fmax)), psd_s_data_log, color = 'blue', label = 'Suppress', linewidth = lw)
    ax.plot(range(fmin, round(fmax)), psd_r_data_log, color = 'green', label = 'Reappraise', linewidth = lw)
    ax.plot(range(fmin, round(fmax)), psd_wp_data_log, color = 'orange', label = 'Watch positive', linewidth = lw)
    ax.plot(range(fmin, round(fmax)), psd_wn_data_log, color = 'black', label = 'Watch neutral', linewidth = lw)

    ## Setting axis
    # Y AXIS
    plt.ylabel('Power spectrum density [log(dB)]', fontsize = fs+2, fontweight='bold') #!!!
    ax.set_ylim([ymin_lg, ymax_lg]) #!!!
    # ax.set_yticks(np.linspace(ymin_lg, ymax_lg, 5))  # exactly 5 tick positions
    # plt.yticks(fontsize=fs)   
    ax.set_yticks(np.round(np.linspace(ymin_lg, ymax_lg, 5), 1))  # 5 ticks, 1 decimal
    plt.yticks(fontsize=fs)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # X AXIS
    plt.xlabel('Frequency (Hz)', fontsize = fs+2, fontweight='bold') #!!!
    ax.set_xlim(left=fmin, right=fmax-1) 
    plt.xticks(np.arange(fmin+3, fmax, step=5))
    plt.xticks(fontsize=fs)   

    # Legend
    ax.legend(fontsize=fs)

    # Specification line
    lower_bound = result[0]['selected_frequency'] - selection_range
    upper_bound = result[0]['selected_frequency'] + selection_range
    ax.axvline(lower_bound, color='black', linestyle='--', linewidth=lw-1)  
    ax.axvline(upper_bound, color='black', linestyle='--', linewidth=lw-1)

    ## Picture management
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(1)  # 1-second delay
    plt.savefig('PSD1.png', dpi=300, bbox_inches='tight')  # Recommended
    fig_psd1.show()


    ###### CHANNEL 2 PLOT
    fig_psd2, ax = plt.subplots(1,1) #, layout='constrained'

    ## Fulfilling the plot
    ax.plot(range(fmin, round(fmax)), psd_s_data_log2, color = 'blue', label = 'Suppress', linewidth = lw)
    ax.plot(range(fmin, round(fmax)), psd_r_data_log2, color = 'green', label = 'Reappraise', linewidth = lw)
    ax.plot(range(fmin, round(fmax)), psd_wp_data_log2, color = 'orange', label = 'Watch positive', linewidth = lw)
    ax.plot(range(fmin, round(fmax)), psd_wn_data_log2, color = 'black', label = 'Watch neutral', linewidth = lw)

    ## Setting axis
    # Y AXIS
    plt.ylabel('Power spectrum density [log(dB)]', fontsize = fs+2, fontweight='bold') #!!!
    ax.set_ylim([ymin_lg, ymax_lg]) #!!!
    # ax.set_yticks(np.linspace(ymin_lg, ymax_lg, 5))  # exactly 5 tick positions
    # plt.yticks(fontsize=fs)   
    ax.set_yticks(np.round(np.linspace(ymin_lg, ymax_lg, 5), 1))  # 5 ticks, 1 decimal
    plt.yticks(fontsize=fs)

    # X AXIS
    plt.xlabel('Frequency (Hz)', fontsize = fs+2, fontweight='bold') #!!!
    ax.set_xlim(left=fmin, right=fmax-1) 
    plt.xticks(np.arange(fmin+3, fmax, step=5))
    plt.xticks(fontsize=fs)   

    # Legend
    ax.legend(fontsize=fs)

    # Specification line
    lower_bound = result[0]['selected_frequency'] - selection_range
    upper_bound = result[0]['selected_frequency'] + selection_range
    ax.axvline(lower_bound, color='black', linestyle='--', linewidth=lw-1)  
    ax.axvline(upper_bound, color='black', linestyle='--', linewidth=lw-1)

    ## Picture management
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(1)  # 1-second delay
    plt.savefig('PSD2.png', dpi=300, bbox_inches='tight')  # Recommended
    fig_psd2.show()
    
    return fig_psd1, fig_psd2

# %% STATISTICAL INFERENCES

def stat_oscillation(stat_filename, area_of_interest_1, area_of_interest_2, which_connection_to_work_with, fs): 
    
    df = pd.read_excel('{}'.format(stat_filename), sheet_name=area_of_interest_1)
    df3 = pd.read_excel('{}'.format(stat_filename), sheet_name=area_of_interest_2)
    
    ### SPLITING
    O_H_lat = df[df['Group']=='Lat']
    I_H_S   = df3
    
    ### Hipocampus - Orbitofrontal area
    which_connection_to_work_with = 'H-O'
    if which_connection_to_work_with == 'H-O':
        what_to_vis = O_H_lat
        print("Hipocampus and Orbitofrontal area")
    if which_connection_to_work_with == 'H-I':
        what_to_vis = O_H_lat 
        print("Hipocampus and Insula")
    
    aaaaa_wn_wp = pg.wilcoxon(what_to_vis['WN'], what_to_vis['WP'], alternative='two-sided')
    aaaaa_wn_s  = pg.wilcoxon(what_to_vis['WN'], what_to_vis['S'],  alternative='two-sided')
    aaaaa_wn_r  = pg.wilcoxon(what_to_vis['WN'], what_to_vis['R'],  alternative='two-sided')
    aaaaa_wp_s  = pg.wilcoxon(what_to_vis['WP'], what_to_vis['S'],  alternative='two-sided')
    aaaaa_wp_r  = pg.wilcoxon(what_to_vis['WP'], what_to_vis['R'],  alternative='two-sided')
    aaaaa_s_r   = pg.wilcoxon(what_to_vis['S'], what_to_vis['R'],   alternative='two-sided')
    
    
    # ###############################################################################
    to_vis = [what_to_vis['WN'], what_to_vis['WP'], what_to_vis['S'], what_to_vis['R']]
    
    #################### SIGNIFICANT LINE
    wn_wp = aaaaa_wn_wp['p-val'][0]
    wn_s  = aaaaa_wn_s['p-val'][0]
    wn_r  = aaaaa_wn_r['p-val'][0]
    wp_s  = aaaaa_wp_s['p-val'][0]
    wp_r  = aaaaa_wp_r['p-val'][0]
    s_r   = aaaaa_s_r['p-val'][0]
    p_values = [wn_wp, wn_s, wn_r, wp_s, wp_r, s_r]

    # Create the figure and subplots
    fig_bxplt, (ax_sig, ax_box) = plt.subplots(
                            2, 1, gridspec_kw={'height_ratios': [0.4, 4]}, figsize=(8, 10))
    
    # Plot the boxplots on the larger subplot
    palette = ['black', 'orange', 'blue', 'green']
    sns.boxplot(data=to_vis,
                palette=palette,
                saturation=1,
                width=0.8,
                dodge=True,
            
                # fliersize=5,
                ax=ax_box, whis=1.5)
    
    for i, artist in enumerate(ax_box.artists):
        if palette[i] == 'black':  # Check if the boxplot color is black
            artist.set_edgecolor('white')  # Change border color to white
            artist.set_linewidth(2)  # Optional: Adjust the border width
            
    sns.stripplot(
        data=to_vis,
        palette=['red', 'red', 'red', 'red'],
        size=7,
        ax=ax_box)
     
    # Remove x-axis labels from the top subplot
    ax_sig.set_xticks([])
    ax_sig.set_yticks([])
        
    # Remove the y-axis and spines from the top subplot for a cleaner look
    # ax_sig.set_yticks([])
    ax_sig.spines['top'].set_visible(False)
    ax_sig.spines['right'].set_visible(False)
    ax_sig.spines['left'].set_visible(False)
    ax_sig.spines['bottom'].set_visible(False)
    
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.05)
    
    black_box_median = np.median(to_vis[0])
    
    # Get the x-coordinate for the black box
    x_black_box = 0  # The first box corresponds to index 0
    
    # Add a horizontal line at the mean height of the black box
    ax_box.hlines(
        y=black_box_median,          # Mean height
        xmin=x_black_box - 0.4,    # Start of the line (slightly left of the box)
        xmax=x_black_box + 0.4,    # End of the line (slightly right of the box)
        color='white',               # Line color
        linewidth=2,               # Line thickness
        linestyle='-',             # Line style
    )
     
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    
    plt.pause(1)  # 1-second delay
    
    # ---------------------------------------------------------------
    ax_box.set_xticklabels(['Watch neutral', 'Watch positive', 'Suppress', 'Reappraise'], fontsize=fs)
    ax_box.set_xlabel('Experimental conditions', fontsize=fs+2, fontweight = 'bold')
    ax_box.set_ylabel('PLV AUC on the peak frequency', fontsize=fs+2, fontweight = 'bold')
    ax_box.set_yticklabels([0, 0.08, 0.16, 0.24, 0.32, 0.4]) #### !!! Problem is here
    ax_box.set_ylim(0, 0.4)                                  #### !!! Problem is here
    ax_box.set_yticks(np.linspace(0, 0.4, 5))                #### !!! Problem is here
    ax_box.tick_params(axis='y', labelsize=fs)

    plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
        
    return wn_wp, wn_s, wn_r, wp_s, wp_r, s_r, fig_bxplt


def figure_2(pilot_data_file_name, sign_level_input, fs):
    ### PILOT RESULTS #########################################################
    df = pd.read_excel('{}'.format(pilot_data_file_name))
    WN = df['WN'].to_numpy()
    WP = df['WP'].to_numpy()
    S  = df['S'].to_numpy()
    R  = df['R'].to_numpy()
    
    # Example data
    to_vis = [WN, WP, S, R]  
    
    #### SIGNIFICANCE LINE
    sign_level = sign_level_input  

    # Create the figure and subplots
    fig_bxplt, (ax_sig, ax_box) = plt.subplots(
                            2, 1, gridspec_kw={'height_ratios': [0.4, 4]}, 
                            figsize=(8, 10))
    
    # Plot the boxplots on the larger subplot
    palette = ['black', 'orange', 'blue', 'green']
    sns.boxplot(data=to_vis,
                palette=palette,
                saturation=1,
                width=0.8,
                dodge=True,
            
                # fliersize=5,
                ax=ax_box, whis=1.5)
    
    for i, artist in enumerate(ax_box.artists):
        if palette[i] == 'black':  # Check if the boxplot color is black
            artist.set_edgecolor('white')  # Change border color to white
            artist.set_linewidth(2)  # Optional: Adjust the border width
            
    sns.stripplot(
        data=to_vis,
        palette=['red', 'red', 'red', 'red'],
        size=7,
        ax=ax_box)
     
    # Customize the boxplot
    ax_box.set_xticklabels(['Watch neutral', 'Watch positive', 'Suppress', 'Reappraise'], fontsize=fs)
    ax_box.set_xlabel('Experimental conditions', fontsize=fs+2, fontweight = 'bold')
    ax_box.set_ylabel('Average ratings of the movies', fontsize=fs+2, fontweight = 'bold')
    ax_box.set_yticks(np.linspace(1, 9, 5))
    ax_box.tick_params(axis='y', labelsize=fs)
    
    # Remove x-axis labels from the top subplot
    ax_sig.set_xticks([])
    ax_sig.set_yticks([])
    
    # Remove the y-axis and spines from the top subplot for a cleaner look
    # ax_sig.set_yticks([])
    ax_sig.spines['top'].set_visible(False)
    ax_sig.spines['right'].set_visible(False)
    ax_sig.spines['left'].set_visible(False)
    ax_sig.spines['bottom'].set_visible(False)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.05)
    
    black_box_median = np.median(to_vis[0])
    
    # Get the x-coordinate for the black box
    x_black_box = 0  # The first box corresponds to index 0
    
    # Add a horizontal line at the mean height of the black box
    ax_box.hlines(
        y=black_box_median,          # Mean height
        xmin=x_black_box - 0.4,    # Start of the line (slightly left of the box)
        xmax=x_black_box + 0.4,    # End of the line (slightly right of the box)
        color='white',               # Line color
        linewidth=2,               # Line thickness
        linestyle='-',             # Line style
    )
    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    
    return fig_bxplt
  
def figure_3(pilot_data_file_name, sign_level_input, fs):
    ### EXPERIMENT RESULTS ####################################################
    df = pd.read_excel('{}'.format(pilot_data_file_name))
    WN = df['WN'].to_numpy()
    WP = df['WP'].to_numpy()
    S  = df['S'].to_numpy()
    R  = df['R'].to_numpy()
    
    # Example data
    to_vis = [WN, WP, S, R]
    
    #### SIGNIFICANCE LINE
    sign_level = sign_level_input
    
    # Create the figure and subplots
    fig_bxplt, (ax_sig, ax_box) = plt.subplots(
                            2, 1, gridspec_kw={'height_ratios': [0.4, 4]}, figsize=(8, 10))
    
    # Plot the boxplots on the larger subplot
    palette = ['black', 'orange', 'blue', 'green']
    sns.boxplot(data=to_vis,
                palette=palette,
                saturation=1,
                width=0.8,
                dodge=True,
            
                # fliersize=5,
                ax=ax_box, whis=1.5)
    
    for i, artist in enumerate(ax_box.artists):
        if palette[i] == 'black':  # Check if the boxplot color is black
            artist.set_edgecolor('white')  # Change border color to white
            artist.set_linewidth(2)  # Optional: Adjust the border width
            
    sns.stripplot(
        data=to_vis,
        palette=['red', 'red', 'red', 'red'],
        size=7,
        ax=ax_box)
     
    # Customize the boxplot
    ax_box.set_xticklabels(['Watch neutral', 'Watch positive', 'Suppress', 'Reappraise'], fontsize=fs)
    ax_box.set_xlabel('Experimental conditions', fontsize=fs+2, fontweight = 'bold')
    ax_box.set_ylabel('Average ratings of the movies', fontsize=fs+2, fontweight = 'bold')
    ax_box.set_yticks(np.linspace(1, 9, 5))
    ax_box.tick_params(axis='y', labelsize=fs)
    
    # Remove x-axis labels from the top subplot
    ax_sig.set_xticks([])
    ax_sig.set_yticks([])
    
    # Remove the y-axis and spines from the top subplot for a cleaner look
    ax_sig.spines['top'].set_visible(False)
    ax_sig.spines['right'].set_visible(False)
    ax_sig.spines['left'].set_visible(False)
    ax_sig.spines['bottom'].set_visible(False)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.05)
    
    black_box_median = np.median(to_vis[0])
    
    # Get the x-coordinate for the black box
    x_black_box = 0  # The first box corresponds to index 0
    
    # Add a horizontal line at the mean height of the black box
    ax_box.hlines(
        y=black_box_median,          # Mean height
        xmin=x_black_box - 0.4,    # Start of the line (slightly left of the box)
        xmax=x_black_box + 0.4,    # End of the line (slightly right of the box)
        color='white',               # Line color
        linewidth=2,               # Line thickness
        linestyle='-',             # Line style
    )
        
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    
    return fig_bxplt

def figure_4(real_subject, fs, ch_name): 

    #### GENERAL FIGURE
    image_files = ['boxplot.png', 'PLV.png', 'PSD1.png', 'PSD2.png'] #!!! Change the picture of the boxplot!!!

    # Make all panels equal-sized; reduce vertical gap between rows
    fig = plt.figure(figsize=(12, 12))
    main_gs = gridspec.GridSpec(
        2, 2, figure=fig,
        wspace=0.15,   # horizontal spacing between left/right panels
        hspace=0.05,   # <-- smaller vertical spacing: top row closer to bottom row
        left=0.05, right=0.95,
        top=0.95, bottom=0.06
    )

    ax1 = fig.add_subplot(main_gs[0, 0])  # Boxplot image
    ax2 = fig.add_subplot(main_gs[0, 1])  # PLV image
    ax3 = fig.add_subplot(main_gs[1, 0])  # PSD1 image
    ax4 = fig.add_subplot(main_gs[1, 1])  # PSD2 image

    # Plot images; keep axes turned off so sizes are governed purely by GridSpec cells
    for ax, img_file in zip([ax1, ax2, ax3, ax4], image_files):
        img = imread(img_file)
        ax.imshow(img, aspect='auto')   # fills the cell; all cells are same size
        ax.axis('off')

    # Slight headroom for the suptitle
    # 2) Top row is already closer (hspace=0.05); we keep modest top margin
    plt.subplots_adjust(top=0.93)

    fig.suptitle(f"Subject {real_subject}", y=0.975, fontsize=16, fontweight='bold')

    plt.savefig('combined_shifted.png', bbox_inches='tight', dpi=300)
    plt.show()

    ### CAPTURE
    left, right = ch_name.split(' - ')[0], ch_name.split(' - ')[1]
    XXX = f"{left} - {right}"
    YYY = left
    ZZZ = right

    caption = (
        f"(B) PLV for all conditions for the best example pair of channels ({XXX}) with the peak frequency area "
        f"highlighted with dashed vertical lines. "
        f"(C–D) PSD for channels {YYY} (C) and {ZZZ} (D), which are displayed for the best example PLV channel."
    )

    print(caption)

    # Slight headroom for the suptitle
    fig.suptitle(f"Subject {real_subject}", y=1, fontsize=16, fontweight='bold')

    # Add panel letters
    ax1.text(-0.01, 1.02, 'A', transform=ax1.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')
    ax2.text(-0.01, 1.02, 'B', transform=ax2.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')
    ax3.text(-0.01, 1.02, 'C', transform=ax3.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')
    ax4.text(-0.01, 1.02, 'D', transform=ax4.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')

  
    fig.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig('combined_shifted.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return fig, caption 

# %% BRAIN VISUALIZATION

def figure_brain(Data_path, coords_hippocampus_lh_input, coords_orbitofrontal_lh_input,
                 color_condition_1, name_1, alpha_input, coords_insula_lh_input, 
                 name_2, color_condition_2):
    
    import pyvista as pv
    fs_dir = '{}/fsaverage'.format(Data_path)
    subjects_dir = fs_dir
    aseg_path = fs_dir + '/mri/aparc+aseg'
    time.sleep(2) 
    
    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)
    time.sleep(2) 
    
    ###### ADD LINE
    coords_hippocampus_lh   = coords_hippocampus_lh_input
    coords_orbitofrontal_lh = coords_orbitofrontal_lh_input
    coords_insula_lh  = coords_insula_lh_input
    
    ###########################################################################
    ################################# CONDITION 1 #############################
    
    # Brain in mm space
    Brain = mne.viz.get_brain_class()
    brain = Brain("fsaverage", "lh", "orig", subjects_dir=subjects_dir,
                  alpha=alpha_input, cortex="low_contrast", background="white", size=(800, 600))
    
    
    brain.add_volume_labels(aseg=aseg_path, labels=["Left-Hippocampus", 
                                                    "ctx-lh-insula",
                                                    "ctx-lh-lateralorbitofrontal"
                                                    ], 
                            colors = ['#9457eb', 'yellow', 'red'], alpha=0.2, legend=False)
    
    # Add lines directly using the plotter
    line_lh = np.array([coords_hippocampus_lh, coords_orbitofrontal_lh])
    line_lh2 = np.array([coords_hippocampus_lh, coords_orbitofrontal_lh])
    pline = pv.Line(line_lh2[0], line_lh2[1])
    tube = pline.tube(radius=2.0)   # <-- increase radius for a thicker tube
    brain.plotter.add_mesh(tube, color=color_condition_1)
      
    brain.show_view('lateral')   # standard left lateral
    brain.save_image("{}.png".format(name_1), mode='rgb')
    time.sleep(2) 
       
    ###########################################################################
    ################################# CONDITION 2 #############################
    
    # Brain in mm space
    Brain = mne.viz.get_brain_class()
    brain = Brain("fsaverage", "lh", "orig", subjects_dir=subjects_dir,
                  alpha=alpha_input, cortex="low_contrast", background="white", size=(800, 600))
        
    brain.add_volume_labels(aseg=aseg_path, labels=["Left-Hippocampus", 
                                                    "ctx-lh-insula",
                                                    "ctx-lh-lateralorbitofrontal"
                                                    ], 
                            colors = ['#9457eb', 'yellow', 'red'], alpha=0.2, legend=False)
    
    
    ###### ADD LINE
    line_lh = np.array([coords_hippocampus_lh, coords_orbitofrontal_lh])
    
    # Add lines directly using the plotter
    import pyvista as pv
    line_lh2 = np.array([coords_hippocampus_lh, coords_orbitofrontal_lh])
    pline = pv.Line(line_lh2[0], line_lh2[1])
    tube = pline.tube(radius=2.0)   # <-- increase radius for a thicker tube
    brain.plotter.add_mesh(tube, color=color_condition_2)
    line_lh3 = np.array([coords_hippocampus_lh, coords_insula_lh])
    pline = pv.Line(line_lh3[0], line_lh3[1])
    tube = pline.tube(radius=2.0)   # <-- increase radius for a thicker tube
    brain.plotter.add_mesh(tube, color=color_condition_2)  
    brain.show_view('lateral')   # standard left lateral
    brain.save_image("{}.png".format(name_2), mode='rgb')
    time.sleep(2) 
     
    ###########################################################################
    ################################# OVERALL #################################
    files = ["{}.png".format(name_1),"{}.png".format(name_2)]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))   # 3 columns × 3 rows
    axes = axes.ravel()                                # flatten to 1-D for easy looping
    
    for ax, fname in zip(axes, files):
        img = mpimg.imread(fname)                      # read the PNG
        ax.imshow(img)                                 # display it
        ax.axis('off')                                 # hide axis ticks & frame
    
    # Optional: remove any unused axes (if you have <9 images)
    for ax in axes[len(files):]:
        ax.axis('off')
    
    plt.tight_layout()                                 # reduce spacing between panels
    plt.savefig("COMBINED.png", dpi=300)          # save the 3×3 grid as a single PNG
    plt.show()
    time.sleep(2) 

    return fig