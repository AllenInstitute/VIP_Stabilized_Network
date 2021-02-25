# -*- coding: utf-8 -*-
"""
Created on 20210221

@author: danielm
"""
import os, sys

import numpy as np
import pandas as pd

import cPickle as pickle
from sync import Dataset
import tifffile as tiff
import matplotlib.pyplot as plt

BLACK = 0
WHITE = 255
GRID_SPACING = 4.65 #degrees
PIXEL_SIZE = 9.3 #degrees

def run_analysis(imaging_duration=3400,#seconds
                 frames_per_sec=30,
                 param_path=''#TODO:define this path for the Deepscope!
                 ):
    
    exptpath = r'C:\Repos\DeepDiveDayOne\401001\new_position\column5\timeseries'
    num_frames = 102000#TODO: enter the actual number of frames!
    
    stim_table = create_stim_table(exptpath,param_path)
    
    fluorescence = get_wholefield_fluorescence(exptpath,num_frames)
    
    mean_sweep_response, sweep_response = get_mean_sweep_response(fluorescence,stim_table,frames_per_sec=frames_per_sec)
    
    condition_response_means, condition_responses = calculate_pixel_responses(mean_sweep_response,sweep_response,stim_table)
    
    plot_sweep_response(condition_response_means,exptpath)

def plot_sweep_response(condition_response_means,stim_table,exptpath):
    
    resp_max = np.max(condition_response_means)
    
    (num_y,num_x,num_colors) = np.shape(condition_response_means)
    origin_x_idx = int(num_x/2) - 1 #indices for pixel at center of monitor
    origin_y_idx = int(num_y/2) - 1
    
    x_degrees = GRID_SPACING*np.arange(num_x)
    x_degrees -= x_degrees[origin_x_idx]
    
    y_degrees = GRID_SPACING*np.arange(num_y)
    y_degrees -= y_degrees[origin_y_idx]
    
    plt.figure(figsize=(20,20))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)     
    
    ax1.imshow(condition_response_means[:,:,0],cmap='RdBu',vmin=-resp_max,vmax=resp_max,interpolation='none',origin='lower')
    ax2.imshow(condition_response_means[:,:,1],cmap='RdBu',vmin=-resp_max,vmax=resp_max,interpolation='none',origin='lower')
    
    ax1.set_title('Black pixels')
    ax2.set_title('White pixels')
    ax1.set_xticks(np.arange(num_x))
    ax1.set_xticklabels([str(round(x,1)) for x in x_degrees])
    ax1.set_yticks(np.arange(num_y))
    ax1.set_yticklabels([str(round(y,1)) for y in y_degrees])
    
    #calculate the centroid of the maps
    rectified_responses = np.where(condition_response_means>0,condition_response_means,0.0)
    summed_resp = rectified_responses.sum()
    x_comp = rectified_responses * x_degrees.reshape(1,num_x)
    y_comp = rectified_responses * y_degrees.reshape(num_y,1)
    centroid_x = round(x_comp.sum() / summed_response,1)
    centroid_y = round(y_comp.sum() / summed_response,1)
    
    ax2.set_xlabel('Population Center X: '+str(centroid_x)+' , Y: '+str(centroid_y))
    
    plt.savefig(exptpath+'/population_RF.png')
    plt.close() 
    
    print('Population Center X: '+str(centroid_x)+' , Y: '+str(centroid_y))

def calculate_pixel_responses(mean_sweep_response,sweep_response,stim_table):
    
    x_pos = np.unique(stim_table['Pixel_X'].values)
    x_pos = x_pos[np.argwhere(np.isfinite(x_pos))]
    y_pos = np.unique(stim_table['Pixel_Y'].values)
    y_pos = y_pos[np.argwhere(np.isfinite(y_pos))]
    
    (num_sweeps,sweeplength) = np.shape(sweep_response)
    
    condition_responses = np.zeros((len(y_pos),len(x_pos),2,sweeplength))
    condition_response_means = np.zeros((len(y_pos),len(x_pos),2))
    for iy,y in enumerate(y_pos):
        is_y = stim_table['Pixel_Y'].values == y
        for ix,x in enumerate(x_pos):
            is_x = stim_table['Pixel_X'].values == x
            for i_color,color in enumerate([BLACK,WHITE]):
                is_color = stim_table['Pixel_Color'].values == color
                is_condition = is_y & is_x & is_color
                condition_sweeps = np.argwhere(is_condition)[:,0]
                condition_responses[iy,ix,i_color] = np.mean(sweep_response[condition_sweeps],axis=0)
                condition_response_means[iy,ix,i_color] = np.mean(mean_sweep_response[condition_sweeps])
                
    return condition_response_means, condition_responses
    
def get_mean_sweep_response(fluorescence,stim_table,frames_per_sec):

    sweeplength = int(stim_table.End[1] - stim_table.Start[1])
    delaylength = int(0.1*frames_per_sec)
    
    num_sweeps = len(stim_table['Start'])
    mean_sweep_response = np.zeros((num_sweeps,))    
    sweep_response = np.zeros((num_sweeps,sweeplength))
    for i in range(num_sweeps):
        response_start = int(stim_table['Start'][i]+delaylength)
        response_end = int(stim_table['Start'][i] + sweeplength + delaylength)
        baseline_start = int(start-sweeplength)
        sweep_f = fluorescence[response_start:response_end]
        baseline_f = fluorescence[baseline_start:response_start]
        sweep_dff = 100*((sweep_f/np.mean(baseline_f))-1)
        sweep_response[i,:] = sweep_f
        mean_sweep_response[i] = np.mean(sweep_dff)
   
    return mean_sweep_response, sweep_response

def load_single_tif(file_path):  
    return tiff.imread(file_path)

def get_wholefield_fluorescence(im_directory,num_frames):  
    
    avg_fluorescence = np.zeros((num_frames,))
    curr_frame = 0
    for f in os.listdir(im_directory):
        if f.endswith('.tif'):
            print "Processing " + f
            this_stack = load_single_tif(im_directory+'/'+f)
            num_stack_frames = np.shape(this_stack)[0]
            for i in range(num_stack_frames):
                if curr_frame<num_frames:
                    avg_fluorescence[curr_frame] = np.mean(this_stack[i,:,:])
                    curr_frame += 1
            this_stack = None
    return avg_fluorescence
    
def create_stim_table(exptpath,param_path):
    
    #load stimulus and sync data
    data = load_pkl(exptpath)
    twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise = load_sync(exptpath)
    
    display_sequence = data['stimuli'][0]['display_sequence']
    display_sequence += data['pre_blank_sec']
    display_sequence *= int(data['fps']) #in stimulus frames
    
    sweep_frames = data['stimuli'][0]['sweep_frames']
    stimulus_table = pd.DataFrame(sweep_frames,columns=('start','end'))            
    stimulus_table['dif'] = stimulus_table['end']-stimulus_table['start']
    stimulus_table.start += display_sequence[0,0]
    for seg in range(len(display_sequence)-1):
        for index, row in stimulus_table.iterrows():
            if row.start >= display_sequence[seg,1]:
                stimulus_table.start[index] = stimulus_table.start[index] - display_sequence[seg,1] + display_sequence[seg+1,0]
    stimulus_table.end = stimulus_table.start+stimulus_table.dif
    print len(stimulus_table)
    stimulus_table = stimulus_table[stimulus_table.end <= display_sequence[-1,1]]
    stimulus_table = stimulus_table[stimulus_table.start <= display_sequence[-1,1]]            
    print len(stimulus_table)
    sync_table = pd.DataFrame(np.column_stack((twop_frames[stimulus_table['start']],twop_frames[stimulus_table['end']])), columns=('Start', 'End'))
           
    #populate stimulus parameters
    sweep_order = data['stimuli'][0]['sweep_order']
    sweep_order =  sweep_order[:len(stimulus_table)]    
          
    sn_params = np.load(savepath+'sparse_noise_sweep_info.npz')
    
    #populate sync_table 
    sync_table['Pixel_Color'] = np.NaN
    sync_table['Pixel_X'] = np.NaN
    sync_table['Pixel_Y'] = np.NaN
    for index in np.arange(len(stimulus_table)):
        if (not np.isnan(stimulus_table['end'][index])) & (sweep_order[index] >= 0):
            sync_table['Pixel_Color'][index] = sn_params['sweep_color'][int(sweep_order[index])]
            sync_table['Pixel_X'][index] = sn_params['sweep_x'][int(sweep_order[index])]
            sync_table['Pixel_Y'][index] = sn_params['sweep_y'][int(sweep_order[index])]
            
    return sync_table
    
def load_sync(exptpath):
    
    #verify that sync file exists in exptpath
    syncMissing = True
    for f in os.listdir(exptpath):
        if f.endswith('.h5'):
            syncpath = os.path.join(exptpath, f)
            syncMissing = False
            print "Sync file:", f
    if syncMissing:
        print "No sync file"
        sys.exit()

    #load the sync data from .h5 and .pkl files
    d = Dataset(syncpath)
    
    #set the appropriate sample frequency
    sample_freq = d.meta_data['ni_daq']['counter_output_freq']
    
    #get sync timing for each channel
    twop_vsync_fall = d.get_falling_edges('2p_vsync')/sample_freq
    stim_vsync_fall = d.get_falling_edges('stim_vsync')[1:]/sample_freq #eliminating the DAQ pulse
    
    photodiode_rise = d.get_rising_edges('stim_photodiode')/sample_freq
    photodiode_fall = d.get_falling_edges('stim_photodiode')/sample_freq
    photodiode_transition = np.union1d(photodiode_rise,photodiode_fall)
    
    #make sure all of the sync data are available
    channels = {'twop_vsync_fall': twop_vsync_fall, 'stim_vsync_fall':stim_vsync_fall, 'photodiode_rise': photodiode_rise}
    channel_test = []    
    for i in channels:
        channel_test.append(any(channels[i]))
        print i + ' syncs : ' + str(len(channels[i]))
    if all(channel_test):
        print "All channels present."
    else:
        print "Not all channels present. Sync test failed."
        sys.exit()        
        
    # find the start of the photodiode 1-second pulses:        
    ptd_transition_diff = np.ediff1d(photodiode_transition)
    is_roughly_one_second = np.abs(ptd_transition_diff-1.0) < 0.016
    first_transition_idx = np.argwhere(is_roughly_one_second)[0,0]
    
    first_transition_time = photodiode_transition[first_transition_idx]
    first_stim_vsync  = stim_vsync_fall[0]
    first_delay = first_transition_time - first_stim_vsync
    print 'delay between first stim_vsync and photodiode: ' + str(first_delay)    
    
    #test and correct for photodiode transition errors
    
    ptd_rise_diff = np.ediff1d(photodiode_rise)
#    short = np.where(np.logical_and(ptd_rise_diff>0.1, ptd_rise_diff<0.3))[0]
#    medium = np.where(np.logical_and(ptd_rise_diff>0.5, ptd_rise_diff<1.5))[0]
#    ptd_start = 2
#    for i in medium:
#        if set(range(i-2,i)) <= set(short):
#            ptd_start = i+1
    ptd_start = first_transition_idx
    ptd_end = np.where(photodiode_rise>stim_vsync_fall.max())[0][0] - 1

#    if ptd_start > 3:
#        print "Photodiode events before stimulus start.  Deleted."
        
    ptd_errors = []
    while any(ptd_rise_diff[ptd_start:ptd_end] < 1.8):
        error_frames = np.where(ptd_rise_diff[ptd_start:ptd_end]<1.8)[0] + ptd_start
        print "Photodiode error detected. Number of frames:", len(error_frames)
        photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
        ptd_errors.append(photodiode_rise[error_frames[-1]])
        ptd_end-=1
        ptd_rise_diff = np.ediff1d(photodiode_rise)
    
    if any(np.abs(first_transition_time-photodiode_rise) < 0.02):#first transition is rise
        first_pulse = np.argwhere(np.abs(first_transition_time-photodiode_rise)<0.02)[0,0]
    else:#first transition is a fall
        first_pulse = np.argwhere(photodiode_rise+0.03>first_transition_time)[0,0]
        
    stim_on_photodiode_idx = 60+120*np.arange(0,ptd_end+1-ptd_start-1,1)
    stim_on_photodiode = stim_vsync_fall[stim_on_photodiode_idx]
    photodiode_on = photodiode_rise[first_pulse + np.arange(0,ptd_end-ptd_start,1)]
    delay_rise = photodiode_on - stim_on_photodiode
    
#    print 'ptd_start: ' + str(ptd_start)
#    print str(ptd_end)    
    
#    plt.figure()
#    plt.plot(stim_on_photodiode[:10],'o')
#    plt.plot(photodiode_on[:10],'.r')
#    plt.show()    
    
    delay = np.mean(delay_rise[:-1])   
    print "monitor delay: " , delay
    
    #adjust stimulus time with monitor delay
    stim_time = stim_vsync_fall + delay
    
    #convert stimulus frames into twop frames
    twop_frames = np.empty((len(stim_time),1))
    acquisition_ends_early = 0
    for i in range(len(stim_time)):
        # crossings = np.nonzero(np.ediff1d(np.sign(twop_vsync_fall - stim_time[i]))>0)
        crossings = np.searchsorted(twop_vsync_fall,stim_time[i],side='left') -1
        if crossings < (len(twop_vsync_fall)-1):
            twop_frames[i] = crossings
        else:
            twop_frames[i:len(stim_time)]=np.NaN
            acquisition_ends_early = 1
            break
            
    if acquisition_ends_early>0:
        print "Acquisition ends before stimulus"
        
    return twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise

def load_pkl(exptpath):
    
    #verify that pkl file exists in exptpath
    logMissing = True
    for f in os.listdir(exptpath):
        if f.endswith('.pkl'):
            logpath = os.path.join(exptpath, f)
            logMissing = False
            print "Stimulus log:", f
    if logMissing:
        print "No pkl file"
        sys.exit()
        
    #load data from pkl file
    f = open(logpath, 'rb')
    data = pickle.load(f)
    f.close()
    
    return data
    
if __name__=='__main__':  
    run_analysis()