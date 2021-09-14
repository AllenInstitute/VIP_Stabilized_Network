# -*- coding: utf-8 -*-
"""
Created on 20210221

@author: danielm
"""
import os, sys

import numpy as np
import pandas as pd

import h5py
import json
import pickle
from sync_py3 import Dataset
import tifffile as tiff
import matplotlib.pyplot as plt
from datetime import datetime

BLACK = 0
WHITE = 255
GRID_SPACING = 4.65 #degrees
PIXEL_SIZE = 9.3 #degrees

def run_analysis(mouse_ID = 123456,
                 exptpath = r'C:\Users\danielm\Desktop\day0_test\\', # r'C:\Repos\DeepDiveDayOne\401001\new_position\column5\timeseries',#'/Users/danielm/Desktop/stimuli/'#
                 param_path='//allen/programs/mindscope/workgroups/vipssn/stimuli/day0_analysis/',
                 output_path=r'C:\\ProgramData\\AIBS_MPE\\script_outputs\\',#'/Users/danielm/Desktop/stimuli/'
                 ):
    
    # Runs the entire analysis end-to-end
    #
    # INPUTS
    # mouse_ID (type: int): unique specimen ID for labeling the output files
    #
    # exptpath (type: str): a path to a directory containing 1) stim pkl file,
    #                        2) sync h5 file, and 3) 2P acquisition movie.
    #
    # param_path (type: str): a path to a directory containing sparse_noise_sweep_info.npz
    #
    # output_path (type: str): a path to a directory to write 1) an RF map png
    #                           and 2) a json containing coordinates to be read
    #                           by the day1_to_9 stimulus script.
    
    stim_table = create_stim_table(exptpath,param_path)

    fluorescence = get_wholefield_fluorescence(exptpath,output_path)
        
    frames_per_sec = get_frame_rate(stim_table) 
    
    mean_sweep_response, sweep_response = get_mean_sweep_response(fluorescence,stim_table,frames_per_sec=frames_per_sec)
    
    #stim_table, mean_sweep_response = test_set()
    
    condition_response_means = calculate_pixel_responses(mean_sweep_response,stim_table)
    
    p_val = test_peak_significance(mean_sweep_response,stim_table)
    if p_val < 0.0001:
        best_x, best_y = calculate_population_RF_coordinates(condition_response_means)
    else:
        best_x, best_y = (0.0,0.0)
    
    plot_RF_maps(condition_response_means,stim_table,best_x,best_y,mouse_ID,output_path)

    write_coordinate_to_json(mouse_ID,best_x,best_y,output_path)

    print('Population Center X: '+str(best_x)+' , Y: '+str(best_y))

def get_frame_rate(stim_table,sweep_duration=0.25):
        
    mean_sweep_frames = np.mean(stim_table['End'].values - stim_table['Start'].values)
    
    return int(np.round(mean_sweep_frames / sweep_duration))

def test_set():
    
    num_sweeps = 12152
    param_path = '/Users/danielm/Desktop/stimuli/day0_analysis/'
    
    sn_params = np.load(param_path+'sparse_noise_sweep_info.npz')
    stim_table = pd.DataFrame(np.zeros((num_sweeps,2)), columns=('Start', 'End'))
    stim_table['Start'] = np.round(np.arange(num_sweeps)*7.5 + 10)
    stim_table['End'] = stim_table['Start'] + 7
    stim_table['Pixel_Color'] = np.array(sn_params['sweep_color'])
    stim_table['Pixel_X'] = np.array(sn_params['sweep_x'])
    stim_table['Pixel_Y'] = np.array(sn_params['sweep_y'])
    
    mean_sweep_response = np.random.normal(0.0,1.0,size=(num_sweeps,))
    condition_to_increase = (np.array(sn_params['sweep_x']) == 10) & (np.array(sn_params['sweep_y']) == 5)
    condition_idx = np.argwhere(condition_to_increase)
    mean_sweep_response[condition_idx] += 3

    return stim_table, mean_sweep_response    

def write_coordinate_to_json(mouse_ID,best_x,best_y,output_path):
    
    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = str(mouse_ID)+'_'+datetime_str+'_coordinates.json'
    
    coordinate_dict = {'target_pos':[best_x,best_y]}
    
    with open(output_path+filename, 'w') as f:
        json.dump(coordinate_dict, f)

def write_coordinate_to_file(mouse_ID,best_x,best_y,output_path):
    
    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = str(mouse_ID)+'_'+datetime_str+'_coordinates.txt'
    
    f = open(output_path+filename,'w')
    f.write('('+str(best_x)+','+str(best_y)+')')
    f.close()

def plot_RF_maps(condition_mat,stim_table,best_x,best_y,mouse_ID,exptpath):
    
    #TODO: calculate a better max and min
    resp_max = np.max(condition_mat)
   
    plt.figure(figsize=(20,20))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)     
    
    plot_single_RF_map(ax1,condition_mat[:,:,0],'Black pixels',resp_max)
    plot_single_RF_map(ax2,condition_mat[:,:,1],'White pixels',resp_max)
    
    ax2.set_xlabel('Population Center X: '+str(best_x)+' , Y: '+str(best_y))
    
    plt.savefig(exptpath+'/population_RF_'+str(mouse_ID)+'.png',dpi=300)
    plt.close() 
    
def plot_single_RF_map(ax,RF_map,title_str,r_max):
    
    x_degrees, y_degrees = get_XY_positions(RF_map)
    
    ax.imshow(RF_map,cmap='RdBu_r',vmin=-r_max,vmax=r_max,interpolation='none',origin='lower')
    ax.set_title(title_str)
    ax.set_xticks(np.arange(len(x_degrees)))
    ax.set_xticklabels([str(round(x,1)) for x in x_degrees])
    ax.set_yticks(np.arange(len(y_degrees)))
    ax.set_yticklabels([str(round(y,1)) for y in y_degrees])   

def test_peak_significance(mean_sweep_response,stim_table,num_shuffles=100000):
    
    num_sweeps = len(stim_table)
    num_conditions = get_number_of_conditions(stim_table)
    sweeps_per_condition = int(num_sweeps/num_conditions)
    
    shuffle_peaks = np.zeros((num_shuffles,))
    for ns in range(num_shuffles):
        shuffle_sweeps = np.random.choice(num_sweeps,size=(num_sweeps,))
        shuffle_sweep_responses = mean_sweep_response[shuffle_sweeps]
        
        shuffle_condition_responses = np.mean(shuffle_sweep_responses.reshape(num_conditions,sweeps_per_condition),axis=1)
        shuffle_peaks[ns] =  shuffle_condition_responses.max()
        
    actual_response_means = calculate_pixel_responses(mean_sweep_response,stim_table)
    actual_peak = actual_response_means.max()
    
    p_val = (shuffle_peaks >= actual_peak).mean()
    
    return p_val
    
def calculate_population_RF_coordinates(condition_response_means):
    
    response_map = condition_response_means.sum(axis=2)
    
    x_degrees, y_degrees = get_XY_positions(condition_response_means)
    
    rectified_responses = np.where(response_map>0,response_map,0.0)
    summed_response = rectified_responses.sum()
    x_comp = rectified_responses * x_degrees.reshape(1,len(x_degrees))
    y_comp = rectified_responses * y_degrees.reshape(len(y_degrees),1)
    centroid_x = round(x_comp.sum() / summed_response,1)
    centroid_y = round(y_comp.sum() / summed_response,1)    
    
    return centroid_x, centroid_y

def get_XY_positions(condition_mat):
    
    num_y = condition_mat.shape[0]
    num_x = condition_mat.shape[1]

    origin_x_idx = int(num_x/2) - 1 #indices for pixel at center of monitor
    origin_y_idx = int(num_y/2) - 1
    
    x_degrees = GRID_SPACING*np.arange(num_x)
    x_degrees -= x_degrees[origin_x_idx]
    
    y_degrees = GRID_SPACING*np.arange(num_y)
    y_degrees -= y_degrees[origin_y_idx]
    
    return x_degrees, y_degrees

def get_number_of_conditions(stim_table):
    x_pos = get_stim_attribute(stim_table, 'Pixel_X')
    y_pos = get_stim_attribute(stim_table, 'Pixel_Y')
    colors = get_stim_attribute(stim_table, 'Pixel_Color')
    return len(x_pos) * len(y_pos) * len(colors)

def calculate_pixel_responses(mean_sweep_response,stim_table):
    
    x_pos = get_stim_attribute(stim_table, 'Pixel_X')
    y_pos = get_stim_attribute(stim_table, 'Pixel_Y')

    condition_response_means = np.zeros((len(y_pos),len(x_pos),2))
    for iy,y in enumerate(y_pos):
        is_y = stim_table['Pixel_Y'].values == y
        for ix,x in enumerate(x_pos):
            is_x = stim_table['Pixel_X'].values == x
            for i_color,color in enumerate([BLACK,WHITE]):
                is_color = stim_table['Pixel_Color'].values == color
                is_condition = is_y & is_x & is_color
                condition_sweeps = np.argwhere(is_condition)[:,0]
                condition_response_means[iy,ix,i_color] = np.mean(mean_sweep_response[condition_sweeps])
                
    return condition_response_means
    
def get_stim_attribute(stim_table,attribute_name):
    attr = np.unique(stim_table[attribute_name].values)
    attr = attr[np.argwhere(np.isfinite(attr))]
    return attr

def get_mean_sweep_response(fluorescence,stim_table,frames_per_sec):

    sweeplength = int(stim_table.End[1] - stim_table.Start[1])
    delaylength = int(0.1*frames_per_sec)
    
    num_sweeps = len(stim_table['Start'])
    mean_sweep_response = np.zeros((num_sweeps,))    
    sweep_response = np.zeros((num_sweeps,sweeplength))
    for i in range(num_sweeps):
        response_start = int(stim_table['Start'][i]+delaylength)
        response_end = int(stim_table['Start'][i] + sweeplength + delaylength)
        baseline_start = int(stim_table['Start'][i]-sweeplength)
        sweep_f = fluorescence[response_start:response_end]
        baseline_f = fluorescence[baseline_start:response_start]
        sweep_dff = 100*((sweep_f/np.nanmean(baseline_f))-1)
        sweep_response[i,:] = sweep_f
        mean_sweep_response[i] = np.nanmean(sweep_dff)
   
    return mean_sweep_response, sweep_response

def load_single_tif(file_path):  
    return tiff.imread(file_path)

def get_wholefield_fluorescence(im_directory,output_path,MAX_FRAMES=200000):  
    
    output_filepath = output_path+'wholefield_fluorescence.npy'
    if os.path.isfile(output_filepath):
        avg_fluorescence = np.load(output_filepath)
    else:
        im_filenames = get_2P_filenames(im_directory)
        num_planes = len(im_filenames)
        
        avg_fluorescence = np.zeros((MAX_FRAMES,))
        
        if num_planes==1:#Bessel
            this_stack = load_single_tif(im_directory+'/'+im_filenames['0'])
            num_stack_frames = np.shape(this_stack)[0]
            for i in range(num_stack_frames):
                avg_fluorescence[i] = np.mean(this_stack[i,:,:])
            avg_fluorescence = avg_fluorescence[:num_stack_frames]
        elif num_planes>1:#Multiplane
            for i_plane in range(num_planes):
                print('Averaging plane '+str(i_plane)+'...')
                if i_plane!=1:
                    f = h5py.File(im_directory+'/'+im_filenames[str(i_plane)])
                    this_stack = np.array(f['data'])
                    num_stack_frames = np.shape(this_stack)[0]
                    for i in range(num_stack_frames):
                        avg_fluorescence[i_plane+i*num_planes] = np.mean(this_stack[i,:,:])
                    f.close()
        np.save(output_filepath,avg_fluorescence)
            
    return avg_fluorescence
   
def get_2P_filenames(im_directory):
    
    #Multiplane
    filenames = {}
    for f in os.listdir(im_directory):
        if f.endswith('.h5') and f.find('plane')>-1:
            plane_number = f[-4]
            filenames[plane_number] = f
    
    #Bessel
    if len(filenames.keys())==0:
        for f in os.listdir(im_directory):
            if f.endswith('timeseries.tiff'):
                filenames['0'] = f
                
    return filenames
    
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
    run_analysis()