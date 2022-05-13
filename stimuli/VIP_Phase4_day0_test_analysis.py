#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:19:48 2021

@author: danielm
"""
import os, sys
import pickle
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from sync_py3 import Dataset

BLACK = 0
WHITE = 255
GRID_SPACING = 4.65 #degrees
PIXEL_SIZE = 9.3 #degrees

def whiteboard():

    savepath = '/Users/danielm/Desktop/stim_sessions/613523_day0/'
    param_path = '/Users/danielm/Desktop/stimuli/day0_analysis/'
    #pkl_name = '210316112528_day0.pkl'

    #data = load_pkl(savepath)
    #print(data.keys())

    #events = get_event_traces(savepath)
    dff = get_dff_traces(savepath)
    print(dff.shape)

    stim_table = create_stim_table(savepath,param_path)
    
    
    stim_table = downsample_stim_table(stim_table)
    print(stim_table)
    
    frames_per_sec = get_frame_rate(stim_table) 
    
    mean_sweep_response, sweep_response = get_mean_sweep_response(dff,stim_table,frames_per_sec,savepath)

    condition_response_means = calculate_pixel_responses(mean_sweep_response,stim_table)

    plot_RFs(condition_response_means,savepath)
   
def downsample_stim_table(stim_table):
    
    stim_table['Start'] = (stim_table.Start.values / 2.0).astype(np.int)
    stim_table['End'] = (stim_table.End.values / 2.0).astype(np.int)
    
    return stim_table
   
def plot_RFs(condition_response_means,savepath):
    
    (num_neurons,num_y,num_x,num_colors) = np.shape(condition_response_means)
    for i_neuron in range(num_neurons):
        plt.figure(figsize=(10,10))
        
        cell_max = np.max(condition_response_means[i_neuron])
        
        ax1 = plt.subplot(211)
        ax1.imshow(condition_response_means[i_neuron,:,:,0],
                   origin='lower',
                   interpolation='none',
                   cmap='RdBu_r',
                   vmin=-0.8*cell_max,
                   vmax=0.8*cell_max)
        
        ax2 = plt.subplot(212) 
        ax2.imshow(condition_response_means[i_neuron,:,:,1],
                   origin='lower',
                   interpolation='none',
                   cmap='RdBu_r',
                   vmin=-0.8*cell_max,
                   vmax=0.8*cell_max)
        
        plt.savefig(savepath+'RF_cell_'+str(i_neuron)+'.png',dpi=300)
        plt.close()

def get_event_traces(exptpath):

    for filename in os.listdir(exptpath):
        if filename.endswith('_event.h5'):
            f = h5py.File(exptpath+filename,'r')
            return np.array(f['events'])
    return None

def get_dff_traces(exptpath):

    for filename in os.listdir(exptpath):
        if filename.endswith('_dff.h5'):
            f = h5py.File(exptpath+filename,'r')
            return np.array(f['data'])
    return None

def calculate_pixel_responses(mean_sweep_response,stim_table):
    
    x_pos = get_stim_attribute(stim_table, 'Pixel_X')
    y_pos = get_stim_attribute(stim_table, 'Pixel_Y')

    num_neurons = mean_sweep_response.shape[0]

    condition_response_means = np.zeros((num_neurons,len(y_pos),len(x_pos),2))
    for iy,y in enumerate(y_pos):
        is_y = stim_table['Pixel_Y'].values == y
        for ix,x in enumerate(x_pos):
            is_x = stim_table['Pixel_X'].values == x
            for i_color,color in enumerate([BLACK,WHITE]):
                is_color = stim_table['Pixel_Color'].values == color
                is_condition = is_y & is_x & is_color
                condition_sweeps = np.argwhere(is_condition)[:,0]
                condition_response_means[:,iy,ix,i_color] = np.mean(mean_sweep_response[:,condition_sweeps],axis=1)
                
    return condition_response_means
    
def get_stim_attribute(stim_table,attribute_name):
    attr = np.unique(stim_table[attribute_name].values)
    attr = attr[np.argwhere(np.isfinite(attr))]
    return attr

def get_mean_sweep_response(dff,stim_table,frames_per_sec,savepath):

    savefile = savepath+'mean_sweep_responses.npy'
    sweep_savefile = savepath+'sweep_responses.npy'
    if os.path.isfile(savefile):
        mean_sweep_response = np.load(savefile)
        sweep_response = np.load(sweep_savefile)
    else:
        sweeplength = int(stim_table.End[1] - stim_table.Start[1])
        delaylength = int(0.1*frames_per_sec)
        
        num_sweeps = len(stim_table['Start'])
        num_neurons = dff.shape[0]
        
        mean_sweep_response = np.zeros((num_neurons,num_sweeps,))    
        sweep_response = np.zeros((num_neurons,num_sweeps,sweeplength))
        for i in range(num_sweeps):
            response_start = int(stim_table['Start'][i]+delaylength)
            response_end = int(stim_table['Start'][i] + sweeplength + delaylength)
            #baseline_start = int(stim_table['Start'][i]-sweeplength)
            sweep_f = dff[:,response_start:response_end]
            #baseline_f = dff[:,baseline_start:response_start]
            #sweep_dff = 100*((sweep_f/np.nanmean(baseline_f))-1)
            sweep_response[:,i,:] = sweep_f
            mean_sweep_response[:,i] = np.nanmean(sweep_f,axis=1)
            
        np.save(savefile,mean_sweep_response)
        np.save(sweep_savefile,sweep_response)
   
    return mean_sweep_response, sweep_response

def get_frame_rate(stim_table,sweep_duration=0.25):
        
    mean_sweep_frames = np.mean(stim_table['End'].values - stim_table['Start'].values)
    
    return int(np.round(mean_sweep_frames / sweep_duration))

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
    print(len(stimulus_table))
    stimulus_table = stimulus_table[stimulus_table.end <= display_sequence[-1,1]]
    stimulus_table = stimulus_table[stimulus_table.start <= display_sequence[-1,1]]            
    print(len(stimulus_table))
    sync_table = pd.DataFrame(np.column_stack((twop_frames[stimulus_table['start']],twop_frames[stimulus_table['end']])), columns=('Start', 'End'))
    #sync_table = pd.DataFrame(np.column_stack((stimulus_table['start'],stimulus_table['end'])), columns=('Start', 'End'))
            
    #populate stimulus parameters
    sweep_order = data['stimuli'][0]['sweep_order']
    sweep_order =  sweep_order[:len(stimulus_table)]    
    
    sn_params = np.load(param_path+'sparse_noise_sweep_info.npz')
    
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
        if f.endswith('_sync.h5'):
            syncpath = os.path.join(exptpath, f)
            syncMissing = False
            print("Sync file: "+ f)
    if syncMissing:
        print("No sync file")
        sys.exit()

    #load the sync data from .h5 and .pkl files
    d = Dataset(syncpath)
    print(d.line_labels)
    
    #set the appropriate sample frequency
    sample_freq = d.meta_data['ni_daq']['counter_output_freq']
    
    #get sync timing for each channel
    twop_vsync_fall = d.get_falling_edges('vsync_2p')/sample_freq
    stim_vsync_fall = d.get_falling_edges('vsync_stim')[1:]/sample_freq #eliminating the DAQ pulse
    
    photodiode_rise = d.get_rising_edges('stim_photodiode')/sample_freq
    photodiode_fall = d.get_falling_edges('stim_photodiode')/sample_freq
    photodiode_transition = np.union1d(photodiode_rise,photodiode_fall)
    
    #make sure all of the sync data are available
    channels = {'twop_vsync_fall': twop_vsync_fall, 'stim_vsync_fall':stim_vsync_fall, 'photodiode_rise': photodiode_rise}
    channel_test = []    
    for i in channels:
        channel_test.append(any(channels[i]))
        print(i + ' syncs : ' + str(len(channels[i])))
    if all(channel_test):
        print("All channels present.")
    else:
        print("Not all channels present. Sync test failed.")
        sys.exit()        
        
    # find the start of the photodiode 1-second pulses:        
    ptd_transition_diff = np.ediff1d(photodiode_transition)
    is_roughly_one_second = np.abs(ptd_transition_diff-1.0) < 0.016
    first_transition_idx = np.argwhere(is_roughly_one_second)[0,0]
    
    first_transition_time = photodiode_transition[first_transition_idx]
    first_stim_vsync  = stim_vsync_fall[0]
    first_delay = first_transition_time - first_stim_vsync
    print('delay between first stim_vsync and photodiode: ' + str(first_delay))
    
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
        print("Photodiode error detected. Number of frames: " + len(error_frames))
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
    print("monitor delay: " + str(delay))
    
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
        print("Acquisition ends before stimulus")
        
    return twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise

def load_pkl(exptpath):
    
    #verify that pkl file exists in exptpath
    logMissing = True
    for f in os.listdir(exptpath):
        if f.endswith('.pkl'):
            logpath = os.path.join(exptpath, f)
            logMissing = False
            print("Stimulus log: " + f)
    if logMissing:
        print("No pkl file")
        sys.exit()
        
    #load data from pkl file
    f = open(logpath, 'rb')
    data = pickle.load(f,encoding='latin1')
    f.close()
    
    return data
    
if __name__=='__main__':  
    whiteboard()