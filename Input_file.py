# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:53:55 2025

@author: Nikita Otstavnov
HSE University
COMMENT: the code should be adapted for particular goal. 
Importantly, in stereo-EEG the channels are customized, thus the code should be controlled manually for particular subject
"""

# %% CHANNEL LIBRARY 
"""
Library was done manually for particular subjects in the sample
"""

eeg_channels1 = {'Left Amygdala'      : [],
                'Left Orbitofrontal' : ["OR'4-OR'5", "OR'5-OR'6", "OR'6-OR'7", "OR'7-OR'8", "OR'8-OR'9", "FP'1-FP'2", "FP'2-FP'3"],
                'Left Hippocampus'   : ["NA'1-NA'2", "NA'2-NA'3",  "NA'3-NA'4", "NA'4-NA'5", "HC'2-HC'3", "HC'3-HC'4", "HC'4-HC'5","HC'5-HC'6"],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HC1-HC2', 'HC2-HC3'],
                'Right Insula'       : []   
                    }

eeg_channels4 = {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : [],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HC2-HC3', 'HT1-HT2', 'HC3-HC4', 'HT2-HT3','HC4-HC5', 'HT5-HT6'],
                'Right Insula'       : ['OF1-OF2','OF2-OF3','OF3-OF4']   
                    }


eeg_channels5 = {'Left Amygdala'     : ["NA'1-NA'2", "NA'2-NA'3", "NA'3-NA'4"],
                'Left Orbitofrontal' : [ "R'2-R'3", "R'3-R'4", "R'4-R'5", "R'5-R'6", "R'6-R'7", "R'7-R'8", "R'8-R'9", "OR'1-OR'2", "OR'2-OR'3", "OR'3-OR'4","OR'4-OR'5", "OR'5-OR'6", "OR'6-OR'7", "OR'7-OR'8", "OR'8-OR'9", "OR'9-OR'10"],
                'Left Hippocampus'   : ["HT'1-HT'2", "HT'2-HT'3", "HT'3-HT'4"],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': ['OR1-OR2', 'OR2-OR3'],
                'Right Hippocampus'  : ['HC1-HC2', 'HC2-HC3', 'HC3-HC4', 'HC4-HC5'],
                'Right Insula'       : []   
                    }

eeg_channels6 = {'Left Amygdala'     : ["NA'2-NA'3", "NA'3-NA'4", "NA'4-NA'5"],
                'Left Orbitofrontal' : ["R'1-R'2", "R'2-R'3", "R'3-R'4", "R'4-R'5", "R'5-R'6", "FP'1-FP'2", "FP'2-FP'3", "FP'3-FP'4"],# "FP'4-FP'5"],
                'Left Hippocampus'   : ["HP'1-HP'2", "HP'2-HP'3", "HP'3-HP'4"],#,  "HP'4-HP'5"],#  "HP'5-HP'6"],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HC1-HC2',  'HC2-HC3',  'HC3-HC4'],
                'Right Insula'       : []   
                    }

eeg_channels7 = {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : [],
                'Left Insula'        : ["T'2-T'3", "H'1-H'2", "H'2-H'3"],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : [],
                'Right Insula'       : ['TT2-TT3', 'TT3-TT4', 'H2-H3', 'H3-H4']   
                    }

eeg_channels8 = {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : ["HT'1-HT'2","HT'4-HT'5"],
                'Left Insula'        : [ "OP'2-OP'3"],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HT4-HT5'],
                'Right Insula'       : ['H2-H3', 'H3-H4', 'OP1-OP2', 'OP2-OP3']   
                    }

eeg_channels10 = {'Left Amygdala'    : [],
                'Left Orbitofrontal' : ["OR'2-OR'3","OR'7-OR'8"],
                'Left Hippocampus'   : ["HC'1-HC'2", "HC'2-HC'3", "HC'3-HC'4",  "HC'4-HC'5"],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': ['R1-R2', 'R2-R3', 'R3-R4', 'R4-R5', 'R5-R6', 'R6-R7',  'R7-R8',  'R8-R9'],
                'Right Hippocampus'  : ['HC1-HC2', 'HC2-HC3', 'HC3-HC4', 'HC4-HC5'],
                'Right Insula'       : []   
                    }

eeg_channels11= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : ["HC'1-HC'2", "HC'2-HC'3", "HC'3-HC'4", "HC'4-HC'5", "HC'5-HC'6"],
                'Left Insula'        : [],
                'Right Amygdala'     : [ 'NA1-NA2', 'NA2-NA3'],  #'NA1-NA2', 'NA2-NA3',, , 'NA3-NA4'
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HT1-HT2',   'HT4-HT5',  'HP3-HP4', 'HP4-HP5'], #  'HP5-HP6' , 'HT2-HT3', 'HT2-HT3', 'HT3-HT4',   
                'Right Insula'       : []   
                    }


eeg_channels12 = {'Left Amygdala'     : [],
                'Left Orbitofrontal' : ["OF'1-OF'2"],
                'Left Hippocampus'   : ["HC'3-HC'4"],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : [],
                'Right Insula'       : []   
                    }


eeg_channels14= {'Left Amygdala'     : ["NA'1-NA'2", "NA'2-NA'3", "NA'3-NA'4"],
                'Left Orbitofrontal' : [ "R'1-R'2", "R'2-R'3", "R'4-R'5", "R'5-R'6", "R'8-R'9", "R'9-R'10", "OR'1-OR'2", "OR'4-OR'5", "OR'5-OR'6", "OR'6-OR'7", "OR'7-OR'8"],
                'Left Hippocampus'   : [],
                'Left Insula'        : ["T'3-T'4","NA'5-NA'6"],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : [],
                'Right Insula'       : []   
                    }

eeg_channels16= {'Left Amygdala'     : ["NA'2-NA'3","NA'3-NA'4","NA'4-NA'5"],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : ["NA'1-NA'2", "HC'3-HC'4"],
                'Left Insula'        : [],
                'Right Amygdala'     : ['NA4-NA5'],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['NA1-NA2', 'NA2-NA3','NA3-NA4'], #,'HT1-HT2', 'HT2-HT3',  'HC1-HC2','HC2-HC3',  'HC3-HC4',
                'Right Insula'       : ['OP3-OP4', 'OF1-OF2']    #'OP2-OP3', 
                    }


eeg_channels17= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : [ "HT'1-HT'2","HT'2-HT'3"],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HT2-HT3'],
                'Right Insula'       : []   
                    }


eeg_channels18= {'Left Amygdala'     : [ "NA'2-NA'3", "NA'3-NA'4",],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : [ "HC'3-HC'4"],
                'Left Insula'        : [],
                'Right Amygdala'     : [ 'NA3-NA4'],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['NA1-NA2', 'NA2-NA3', 'HT2-HT3','HT3-HT4', 'HT4-HT5'],
                'Right Insula'       : []   
                    }

eeg_channels20= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : ["HC'1-HC'2",  "HC'2-HC'3", "HC'3-HC'4"],
                'Left Insula'        : ["T'2-T'3"],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HC1-HC2', 'HC2-HC3', 'HC3-HC4', 'HC4-HC5', 'HT1-HT2', 'HT2-HT3','HT3-HT4', 'HT4-HT5'],
                'Right Insula'       : ['OF1-OF2',  'H2-H3',  'O1-O2', 'O2-O3']   
                    }

eeg_channels21= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : ["HT'1-HT'2","HT'2-HT'3","HT'3-HT'4","HT'4-HT'5"],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HT1-HT2','HT2-HT3','HT3-HT4', 'HT4-HT5'],
                'Right Insula'       : []   
                    }


eeg_channels22= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : [ "HT'1-HT'2", "HT'2-HT'3", "HT'3-HT'4","HT'4-HT'5"],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': ['OR2-OR3', 'OR3-OR4', 'OR4-OR5','OR5-OR6'],
                'Right Hippocampus'  : ['HT1-HT2'],
                'Right Insula'       : []   
                    }

eeg_channels23= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : [],
                'Left Insula'        : [ "OF'1-OF'2", "OF'2-OF'3"],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [], #'OF1-OF2'
                'Right Hippocampus'  : ["PFG'1-PFG'2"],
                'Right Insula'       : ['OF2-OF3','OF3-OF4','OF4-OF5','OF5-OF6','OC3-OC4']   
                    }

eeg_channels24= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : [],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': ['OF3-OF4','OF5-OF6'],
                'Right Hippocampus'  : [],
                'Right Insula'       : [ 'CP3-CP4']   
                    }

eeg_channels25= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : [],
                'Left Insula'        : [],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HT1-HT2'],
                'Right Insula'       : []   
                    }

eeg_channels26= {'Left Amygdala'     : [],
                'Left Orbitofrontal' : [],
                'Left Hippocampus'   : ["HC'2-HC'3","HC'3-HC'4"],
                'Left Insula'        : ["OC'2-OC'3"],
                'Right Amygdala'     : [],
                'Right Orbitofrontal': [],
                'Right Hippocampus'  : ['HT1-HT2'],
                'Right Insula'       : [ 'OC1-OC2','OC2-OC3','OC3-OC4','OF5-OF6','OP1-OP2','OP2-OP3','OP3-OP4', 'OP4-OP5',  'OP5-OP6']   
                    }

subject_ductionary = {
                        'sub1': eeg_channels1, 
                        'sub4': eeg_channels4, 
                        'sub5': eeg_channels5, 
                        'sub6': eeg_channels6, 
                        'sub7': eeg_channels7, 
                        'sub8': eeg_channels8, 
                        'sub10': eeg_channels10, 
                        'sub11': eeg_channels11, 
                        'sub12': eeg_channels12, 
                        'sub14': eeg_channels14, 
                        'sub16': eeg_channels16, 
                        'sub17': eeg_channels17, 
                        'sub18': eeg_channels18, 
                        'sub20': eeg_channels20, 
                        'sub21': eeg_channels21, 
                        'sub22': eeg_channels22, 
                        'sub23': eeg_channels23, 
                        'sub24': eeg_channels24, 
                        'sub25': eeg_channels25, 
                        'sub26': eeg_channels26 
                        }



# %% LIBRARIES AND FUNCTIONS

from Function_file import epochs_creation, psd_calculations, psd_visualizer, plv_calculator
from Function_file import calculate_auc_at_selected_frequencies, psd_for_combined_figure
from Function_file import stat_oscillation
from Function_file import figure_2, figure_3, figure_4, figure_brain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% INPUT DATA

Data_path                  = 'K:/EXP_DATA/iEEG_data' #!!! Write you files location here
disk                       = 'K'
subject                    = '5'
final_name                 = '_upd2'
annotation_final_name      = '_updated'

# %% EPOCHING
"""
The data for different subjects can have a different electrode montage with different naming. 
Adapt this part of the code in Function_file particularly for each subject.
"""
tmin_input                 = 0
tmax_input                 = 2
movie_duration_input       = 60
window_for_visualization   = 20
scalings_for_visualization = dict(eeg=216.695e-6)
epochs_list_long_events, raw_new = epochs_creation(Data_path, subject, final_name, annotation_final_name, 
                                                   subject_ductionary, tmin_input, tmax_input, 
                                                   movie_duration_input, window_for_visualization,
                                                   scalings_for_visualization)

epochs_wp = epochs_list_long_events[0]
epochs_s  = epochs_list_long_events[1]
epochs_r  = epochs_list_long_events[2] 
epochs_wn = epochs_list_long_events[3] 

# %% EPOCHING

### CALCULATION ###############################################################
n_jobs        = 4  
fmin          = 2 
fmax          = 80
method        = 'welch'
verbose_input = None
proj_input    = False
tmin_input    = None
tmax_input    = None
average_input = True

wp_psd, wn_psd, r_psd, s_psd, epochs_suppress = psd_calculations(n_jobs, fmin, fmax, method, 
                                                                 epochs_list_long_events, 
                                                                 verbose_input, proj_input, 
                                                                 tmin_input, tmax_input, 
                                                                 average_input)

### VISUALIZATION #############################################################
fmin        = 2 
fmax        = 46
ch          = "HC1-HC2"
fig, ax     = psd_visualizer(fmin, fmax, ch, wp_psd, wn_psd, r_psd, s_psd)

# %% PHASE LOCKING VALUE 

### PLV calculation ###########################################################
fmin        = 0
fmax        = 50
n_jobs      = -1
method_input='plv'
faverage_input = False
verbose_input = None
fskip_input = 0
what_to_take_from_the_dict = ['Left Orbitofrontal', 'Right Orbitofrontal', 'Left Hippocampus', 'Right Hippocampus',  
                               'Left Insula',  'Right Insula', 'Left Amygdala', 'Right Amygdala']

outcome_wn, outcome_wp, outcome_r, outcome_s, index_to_channel, mean_wneu, mean_wpos, mean_reas, mean_supr = plv_calculator(subject, subject_ductionary, raw_new, fmin, fmax, n_jobs, what_to_take_from_the_dict, 
                                                                                                                            epochs_s, epochs_r, epochs_wp, epochs_wn, epochs_suppress, method_input,
                                                                                                                            faverage_input, verbose_input, fskip_input)
# %% AREA UNDER THE CURVE CALCULATION

####### DICTINARY WITH CHANNEL NAMES ##########################################

"""
From the channel list one must chose the most characteristic channel for visualization.
One must choose the channel, but refer to it by the index!
"""
n1_l = outcome_s.indices[0]
n2_l = outcome_s.indices[1]
channel_dict = {}

for j in range(len(n1_l)):
    n1   = n1_l[j] 
    n2   = n2_l[j] 
    ch1_name  = index_to_channel[n1]
    ch2_name  = index_to_channel[n2]
    combined_channels = f"{ch1_name} - {ch2_name}"
    channel_dict[j] = combined_channels

print(channel_dict) 

####### AUC manual selection and calculation ##################################
lw = 4 
selection_range = 0.5
fs = 25
j  = 15 # !!! index of the channel to plot #34 - for sub 20, 15 for sub 5
print(channel_dict[j])
fmin_to_plot = 2 

frequencies = np.array(outcome_r.freqs)  # Your frequency array
plv_watch_neutral = mean_wneu[j,:]
plv_watch_positive = mean_wpos[j,:]
plv_suppress = mean_supr[j,:]
plv_reassess = mean_reas[j,:]

result, fig_plv, fmin, fmax = calculate_auc_at_selected_frequencies(frequencies, plv_watch_neutral, 
                                                                    plv_watch_positive, plv_suppress, 
                                                                    plv_reassess, selection_range, 
                                                                    j, lw, fs, fmin_to_plot)

ch_name = '{} - {} - INDEX_{}'.format(ch1_name , ch2_name, j)
print(ch_name)
print(result)

resulting_auc = pd.DataFrame(result)      

# %% FIGURES 2-4 (5)

###### FIGURE PSD #############################################################
method_psd_input      = 'welch'
n_jobs      = 4  
fmin        = 2 
proj_input    = False
tmin_input    = None
tmax_input    = None

fig_psd1, fig_psd2 = psd_for_combined_figure(ch_name, method_psd_input, fmin, fmax, epochs_list_long_events, verbose_input,
                                            tmin_input, tmax_input, proj_input, lw, n_jobs, fs, result, selection_range)


###### OSCILLATORY STATISTICS #################################################
"""
1. Customize the code in Function_file depending on particular style
2. The significance lines are drawn manually
"""

stat_filename = 'PLV for plotting and statistics 6.xlsx'
area_of_interest_1 ='Orbital'
area_of_interest_2 ='Insula'
which_connection_to_work_with = 'H-I' #!!! "H-O" 

wn_wp, wn_s, wn_r, wp_s, wp_r, s_r, fig_bxplt = stat_oscillation(stat_filename, area_of_interest_1, area_of_interest_2, which_connection_to_work_with, fs)

print('wn vs wp: ', wn_wp)
print('wn vs s: ', wn_s)
print('wn vs r: ', wn_r)
print('wp vs s: ', wp_s)
print('wp vs r: ', wp_r)
print('s vs r: ', s_r)

###### PILOT BEHAVIORAL STATISTICS ############################################
# FIGURE 2 - pilot results
pilot_data_file_name = 'sEEG_pilot_behavioral.xlsx' 
sign_level_input = 0.05 / 6    
figure2 =  figure_2(pilot_data_file_name, sign_level_input, fs)

###### EXP BEHAVIORAL STATISTICS ##############################################
# FIGURE 3 - experimental results
pilot_data_file_name = 'sEEG_experiment_behavioral.xlsx' 
sign_level_input = 0.05 / 6    
figure2 =  figure_3(pilot_data_file_name, sign_level_input, fs)

###### FIGURE 4 (and optionally 5) ############################################
### SUBJECT mapping
mapping_pairs = [
    (1, 1),
    (5, 2),
    (6, 3),
    (10, 4),
    (22, 5),
    (16, 6),
    (11, 7),
    (18, 8),
    (5, 9),
    (6, 10),
    (4, 11),
    (8, 12),
    (20, 13),
    (26, 14),
    (23, 15),
    (16, 16)
]

subject_map = {}
for k, v in mapping_pairs:
    subject_map[k] = v  # last one wins

subject = int(subject)
real_subject = subject_map.get(subject)

# Figure 4 drawing 
fig, caption = figure_4(real_subject, fs, ch_name)

# %% BRAIN VISUALIZATION (Figure 6)

coords_hippocampus_lh_input   = [-24, -20, -20]  # Left Hippocampus
coords_orbitofrontal_lh_input = [-24, 34, -16] # Left Orbitofrontal Cortex
coords_insula_lh_input        = [-34, 5, 0] # Left Orbitofrontal Cortex
color_condition_1 = 'green'
color_condition_2 = 'blue'
name_1 = 'reap'
name_2 = 'suppr'
alpha_input = 0.1

fig = figure_brain(Data_path, coords_hippocampus_lh_input, coords_orbitofrontal_lh_input,
                   color_condition_1, name_1, alpha_input, coords_insula_lh_input, 
                   name_2, color_condition_2)