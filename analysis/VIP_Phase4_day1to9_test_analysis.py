#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:26:45 2021

@author: danielm
"""

import os, sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from StimTable import stim_table as st
from chisq import chisq_categorical as chi

def main():
    
    datapath = '/Users/danielm/Desktop/stim_sessions/'
    
    #load DataFrame that describes all sessions in the dataset
    session_df = get_session_df()

    #run whole analysis for each session:
    for index, row in session_df.iterrows():
        MID = int(row['mouse_ID'])
        day = int(row['Day'])
        session_ID = int(row['session_ID'])
        
        #the analysis assumes that the stim.pkl, sync.h5 and dff.h5 files 
        #are all located in the same directory, called exptpath
        exptpath = to_exptpath(datapath,
                               row['cre_line'],
                               MID,
                               day
                               )
        
        print('Running analysis for MID '+str(MID)+' day '+str(day))
        
        whole_trick(exptpath,session_ID)
    
def whole_trick(exptpath,session_ID):
    
    #check that stim.pkl file exists in the exptpath; uses assert
    exptpath_has_stim_pkl(exptpath,session_ID)
    
    #define where to save outputs; for now, just put them in the exptpath
    savepath = exptpath
    
    #get dict of sweep_tables, one for sizeXcontrast and one for image flashes
    sweep_table = get_sweep_table_dict(session_ID,exptpath,savepath)

    #frame rate can be determined directly from the stim tables
    frame_rate = get_frame_rate(sweep_table['size_by_contrast'])
    print('frame rate: '+str(frame_rate))
    
    #using dF/F until time constant is fixed for these 2X downsampled timeseries
    dff = load_dff_traces(exptpath)
    print(np.shape(dff))
    
    #mean_sweep_responses has shape (num_neurons,num_sweeps)
    size_by_contrast_mse_dff = get_mean_sweep_response(dff,
                                                        sweep_table['size_by_contrast'],
                                                        frame_rate,
                                                        savepath,
                                                        'SizeByContrast',
                                                        'dff')
    
    #find cells with significant tuning across size and contrast conditions via bootstrapped Chi-sq test
    p_vals = chi_square_all_conditions(sweep_table['size_by_contrast'],
                                        size_by_contrast_mse_dff.T,
                                        session_ID,
                                        savepath)    

    #take simple average (mean) across sweeps of each stimulus condition
    condition_responses, blank_sweep_responses = compute_mean_condition_responses(sweep_table['size_by_contrast'],
                                                                                  size_by_contrast_mse_dff.T)
    
    #plot average tuning curve across all neurons 
    #    (can update to only include significantly-tuned neurons: e.g., p_vals<SIG_THRESH)
    plot_population_tuning(condition_responses,blank_sweep_responses,savepath)
    
    #same for each neuron's tuning curve
    plot_single_cell_tuning(condition_responses,
                            blank_sweep_responses,
                            p_vals,
                            savepath)
    
def exptpath_has_stim_pkl(exptpath,session_ID):
    assert os.path.isfile(exptpath+str(session_ID)+'_stim.pkl')
    
def plot_population_tuning(condition_responses,blank_responses,savepath):
    
    fig_filepath = savepath+'population_tuning.png'
    if not os.path.isfile(fig_filepath):
    
        num_neurons = condition_responses.shape[0]
        directions, contrasts, sizes = grating_params()
        plt.figure(figsize=(10,10))
            
        for nc in range(num_neurons):
            condition_responses[nc] = condition_responses[nc] - blank_responses[nc]
            
        #blank-out stimulus conditions that were not presented
        condition_responses[:,:,0,0] = 0
        condition_responses[:,:,2,0] = 0
        condition_responses[:,:,4,0] = 0
        condition_responses[:,:,0,1] = 0
        condition_responses[:,:,2,1] = 0
        condition_responses[:,:,4,1] = 0
        
        population_responses = np.mean(condition_responses,axis=0)
        
        for i_size,size in enumerate(sizes):
            
            ax = plt.subplot(3,1,i_size+1)
            ax.imshow(population_responses[:,:,i_size],
                      cmap='RdBu_r',
                      vmin=-np.max(population_responses),
                      vmax=np.max(population_responses),
                      interpolation='none')
            
            ax.set_xticks(np.arange(len(contrasts)))
            ax.set_xticklabels([str(int(100*x)) + '%' for x in contrasts])
            ax.set_yticks(np.arange(len(directions)))
            ax.set_yticklabels([str(int(x)) for x in directions])
            
            ax.set_ylabel('Direction')
            ax.set_title('Size: '+str(size))
            
            if i_size==2:
                ax.set_xlabel('Contrast')
        
        plt.savefig(fig_filepath,dpi=300)
        plt.close()
        
def plot_single_cell_tuning(condition_responses,blank_responses,p_vals,savepath):
    
    num_neurons = condition_responses.shape[0]
    directions, contrasts, sizes = grating_params()
    
    for nc in range(num_neurons):
        
        fig_filepath = savepath+'tuning_'+str(nc)+'.png'
        if not os.path.isfile(fig_filepath):
        
            plt.figure(figsize=(10,10))
            
            cell_responses = condition_responses[nc] - blank_responses[nc]
            cell_responses[:,0,0] = 0
            cell_responses[:,2,0] = 0
            cell_responses[:,4,0] = 0
            cell_responses[:,0,1] = 0
            cell_responses[:,2,1] = 0
            cell_responses[:,4,1] = 0
            
            for i_size,size in enumerate(sizes):
                
                
                
                ax = plt.subplot(3,1,i_size+1)
                ax.imshow(cell_responses[:,:,i_size],
                          cmap='RdBu_r',
                          vmin=-np.max(cell_responses),
                          vmax=np.max(cell_responses),
                          interpolation='none')
                
                ax.set_xticks(np.arange(len(contrasts)))
                ax.set_xticklabels([str(int(100*x)) + '%' for x in contrasts])
                ax.set_yticks(np.arange(len(directions)))
                ax.set_yticklabels([str(int(x)) for x in directions])
                
                if i_size==2:
                 ax.set_xlabel('Contrast')
                ax.set_ylabel('Direction')
                
                if i_size==0:
                    ax.set_title('Size: '+str(size)+' p_val: '+str(p_vals[nc]))
                else:
                    ax.set_title('Size: '+str(size))
            
            plt.savefig(fig_filepath,dpi=300)
            plt.close()

def load_dff_traces(exptpath):
    
    for f in os.listdir(exptpath):
        if f.find('_dff.h5') > -1:
            dff = np.array(h5py.File(exptpath+f,'r')['data'])
            
            #remove neurons with NaNs; these will get removed at event detection
            # so this keeps dF/F consistent with future event timeseries
            no_nans = np.argwhere(np.sum(np.isnan(dff),axis=1)==0)[:,0]
            return dff[no_nans]
    return None

def get_frame_rate(stim_table,sweep_duration=2.0):
        
    mean_sweep_frames = np.mean(stim_table['End'].values - stim_table['Start'].values)
    
    return int(np.round(mean_sweep_frames / sweep_duration))

def get_mean_sweep_response(dff,
                            stim_table,
                            frames_per_sec,
                            savepath,
                            stim_name,
                            response_type,
                            baseline_subtract=True):

    savefile = savepath+stim_name+'_'+response_type+'_mean_sweep_responses.npy'
    sweep_savefile = savepath+stim_name+'_'+response_type+'_sweep_responses.npy'
    if os.path.isfile(savefile):
        mean_sweep_response = np.load(savefile)
        sweep_response = np.load(sweep_savefile)
    else:
        sweeplength = int(stim_table.End[1] - stim_table.Start[1])
        delaylength = int(0.1*frames_per_sec)
        baseline_length = int(0.5*frames_per_sec)
        
        num_sweeps = len(stim_table['Start'])
        num_neurons = dff.shape[0]
        
        mean_sweep_response = np.zeros((num_neurons,num_sweeps,))    
        sweep_response = np.zeros((num_neurons,num_sweeps,sweeplength))
        for i in range(num_sweeps):
            response_start = int(stim_table['Start'][i]+delaylength)
            response_end = int(stim_table['Start'][i] + sweeplength + delaylength)
            sweep_f = dff[:,response_start:response_end]

            if baseline_subtract:
                baseline_start = int(stim_table['Start'][i]-baseline_length)
                baseline_f = np.nanmean(dff[:,baseline_start:response_start],axis=1).reshape(num_neurons,1)
                sweep_resp = sweep_f - baseline_f
            else:
                sweep_resp = sweep_f
                
            sweep_response[:,i,:] = sweep_resp
            mean_sweep_response[:,i] = np.nanmean(sweep_resp,axis=1)
            
        np.save(savefile,mean_sweep_response)
        np.save(sweep_savefile,sweep_response)
   
    return mean_sweep_response
    
def chi_square_all_conditions(sweep_table,
                              mean_sweep_events,
                              session_ID,
                              savepath):
    
    if os.path.isfile(savepath+str(session_ID)+'_chisq_all.npy'):
        p_vals = np.load(savepath+str(session_ID)+'_chisq_all.npy')
    else:
        p_vals = chi.chisq_from_stim_table(sweep_table,
                                           ['Ori','Size','Contrast'],
                                           mean_sweep_events)
        
        np.save(savepath+str(session_ID)+'_chisq_all.npy',p_vals)
    
    return p_vals

def compute_mean_condition_responses(sweep_table,mean_sweep_events):
    
    (num_sweeps,num_cells) = np.shape(mean_sweep_events) 
    
    directions, contrasts, sizes = grating_params()
    
    condition_responses = np.zeros((num_cells,len(directions),len(contrasts),len(sizes)))
    for i_dir,direction in enumerate(directions):
        is_direction = sweep_table['Ori'] == direction
        for i_contrast,contrast in enumerate(contrasts):
            is_contrast = sweep_table['Contrast'] == contrast
            for i_size,size in enumerate(sizes):
                is_size = sweep_table['Size'] == size
                is_condition = (is_direction & is_contrast & is_size).values
            
                if is_condition.sum()>0:
                    condition_responses[:,i_dir,i_contrast,i_size] = np.mean(mean_sweep_events[is_condition],axis=0)
            
    is_blank = np.isnan(sweep_table['Ori'].values)
    blank_sweep_responses = np.mean(mean_sweep_events[is_blank],axis=0)
            
    return condition_responses, blank_sweep_responses    

def compute_blank_subtracted_NLL(session_ID,savepath,num_shuffles=200000):
    
    #not currently used, but useful to see how robust is the tuning
    #calculates the percentile of the mean response of each neuron to each 
    #stimulus condition in a bootstrapped distribution, then takes a log-tranform
    #of the percentile (centering zero to be the median).
    
    if os.path.isfile(savepath+str(session_ID)+'_blank_subtracted_NLL.npy'):
        condition_NLL = np.load(savepath+str(session_ID)+'_blank_subtracted_NLL.npy')
        blank_NLL = np.load(savepath+str(session_ID)+'_blank_subtracted_blank_NLL.npy')
    else:
        sweep_table = load_sweep_table(savepath,session_ID)
        mean_sweep_events = load_mean_sweep_events(savepath,session_ID)
        
        (num_sweeps,num_cells) = np.shape(mean_sweep_events) 
        directions, SFs, TFs = grating_params()
        
        trials_per_condition, num_blanks = compute_trials_per_condition(sweep_table)
        condition_responses_all, blank_sweep_responses = compute_mean_condition_responses(sweep_table,mean_sweep_events)
        
        condition_NLL = np.zeros((num_cells,len(directions),len(SFs),len(TFs)))
        blank_NLL = np.zeros((num_cells,))
        for i_TF in range(len(TFs)):
            condition_responses = condition_responses_all[:,:,:,i_TF]
            condition_responses = np.swapaxes(condition_responses,0,2)
            condition_responses = np.swapaxes(condition_responses,0,1)
            
            # different conditions can have different number of trials...
            unique_trial_counts = np.unique(trials_per_condition[:,:,i_TF].flatten())
            
            trial_count_mat = np.tile(trials_per_condition[:,:,i_TF],reps=(num_cells,1,1))
            trial_count_mat = np.swapaxes(trial_count_mat,0,2)
            trial_count_mat = np.swapaxes(trial_count_mat,0,1)
            
            blank_shuffle_sweeps = np.random.choice(num_sweeps,size=(num_shuffles*num_blanks,))
            blank_shuffle_responses = mean_sweep_events[blank_shuffle_sweeps].reshape(num_shuffles,num_blanks,num_cells)
            blank_null_dist = blank_shuffle_responses.mean(axis=1)
            
            condition_NLL_one_TF = np.zeros((len(directions),len(SFs),num_cells))
            for trial_count in unique_trial_counts:
                
                #create null distribution and compute condition NLL
                shuffle_sweeps = np.random.choice(num_sweeps,size=(num_shuffles*trial_count,))
                shuffle_responses = mean_sweep_events[shuffle_sweeps].reshape(num_shuffles,trial_count,num_cells)
                
                null_diff_dist = shuffle_responses.mean(axis=1) - blank_null_dist
                actual_diffs = condition_responses.reshape(len(directions),len(SFs),1,num_cells) - blank_sweep_responses.reshape(1,1,1,num_cells)
                resp_above_null = null_diff_dist.reshape(1,1,num_shuffles,num_cells) < actual_diffs
                percentile = resp_above_null.mean(axis=2)
                NLL = percentile_to_NLL(percentile,num_shuffles)
            
                has_count = trial_count_mat == trial_count
                condition_NLL_one_TF = np.where(has_count,NLL,condition_NLL_one_TF)
                
#            #repeat for blank sweeps
#            blank_null_dist_2 = blank_null_dist[np.random.choice(num_shuffles,size=num_shuffles),:]
#            blank_null_diff_dist = blank_null_dist_2 - blank_null_dist
#            actual_diffs = 0.0
#            resp_above_null = blank_null_diff_dist < actual_diffs
#            percentile = resp_above_null.mean(axis=0)
#            blank_NLL = percentile_to_NLL(percentile,num_shuffles)
        
            condition_NLL_one_TF = np.swapaxes(condition_NLL_one_TF,0,2)
            condition_NLL_one_TF = np.swapaxes(condition_NLL_one_TF,1,2)
            condition_NLL[:,:,:,i_TF] = condition_NLL_one_TF
        
        np.save(savepath+str(session_ID)+'_blank_subtracted_NLL.npy',condition_NLL)
        np.save(savepath+str(session_ID)+'_blank_subtracted_blank_NLL.npy',blank_NLL)
        
    return condition_NLL, blank_NLL
    
def percentile_to_NLL(percentile,num_shuffles):
    
    percentile = np.where(percentile==0.0,1.0/num_shuffles,percentile)
    percentile = np.where(percentile==1.0,1.0-1.0/num_shuffles,percentile)
    NLL = np.where(percentile<0.5,
                   np.log10(percentile)-np.log10(0.5),
                   -np.log10(1.0-percentile)+np.log10(0.5))
    
    return NLL

def NLL_to_percentile(NLL):
    
    percentile = np.where(NLL<0.0,
                          10.0**(NLL+np.log10(0.5)),
                          1.0-10.0**(np.log10(0.5)-NLL))
    
    return percentile

def compute_trials_per_condition(sweep_table):
    
    directions, SFs, TFs = grating_params()
    trials_per_condition = np.zeros((len(directions),len(SFs),len(TFs)),dtype=np.int)
    for i_dir,direction in enumerate(directions):
        is_direction = sweep_table['Ori'] == direction
        for i_SF,SF in enumerate(SFs):
            is_SF = sweep_table['SF'] == SF
            for i_TF,TF in enumerate(TFs):
                is_TF = sweep_table['TF'] == TF
                is_condition = (is_direction & is_SF & is_TF).values
                trials_per_condition[i_dir,i_SF,i_TF] = is_condition.sum()

    num_blanks = np.isnan(sweep_table['Ori'].values).sum()

    return trials_per_condition, num_blanks

def grating_params():
    directions = np.arange(0,360,90)
    contrasts = [0.025,0.05,0.1,0.2,0.4,0.8]
    sizes = [12.5,25,250]
    
    return directions, contrasts, sizes
 
def get_sweep_table_dict(session_ID,exptpath,savepath):
    
    filepath = savepath + str(session_ID) + '_sweep_table.h5'
    if os.path.isfile(filepath):
        sweep_table = {}
        sweep_table['size_by_contrast'] = pd.read_hdf(filepath,key='size_by_contrast',mode='r')
        sweep_table['visual_behavior_flashes'] = pd.read_hdf(filepath,key='visual_behavior_flashes',mode='r')
    else:
    
        sweep_table = st.SizeByContrast_tables(exptpath)
    
        sweep_table['size_by_contrast'] = sweep_table_to_downsampled(sweep_table['size_by_contrast'])
        sweep_table['visual_behavior_flashes'] = sweep_table_to_downsampled(sweep_table['visual_behavior_flashes'])
    
        sweep_table['size_by_contrast'].to_hdf(filepath,key='size_by_contrast')
        sweep_table['visual_behavior_flashes'].to_hdf(filepath,key='visual_behavior_flashes')
        
    print(sweep_table)
    
    return sweep_table
 
def sweep_table_to_downsampled(sweep_table,num_frames_averaged=2.0):
    #returns sweep tables after accounting for the 2x downsampling in time done before LIMS upload.
    #2P movies were too long for motion correction algorithm
    
    sweep_table['Start'] = sweep_table['Start'].values / num_frames_averaged
    sweep_table['End'] = sweep_table['End'].values / num_frames_averaged
    
    return sweep_table

def get_session_df():
    #returns list of all sessions in the dataset
    
    #                session_ID, cre, mouseID, day
    session_info = [(1143565396,'Vip',598130,3),
                    (1143779068,'Vip',598130,4),
                    (1144578662,'Vip',598130,5),
                    (1144819429,'Vip',598130,6),
                    
                    (1144953459,'Vip',594263,6),
                    
                    (1167563391,'Vip',616082,1),
                    (1167772447,'Vip',616082,2),
                    (1168059486,'Vip',616082,3),
                    (1168585365,'Vip',616082,4),
                    (1168874989,'Vip',616082,5),
                    (1169157869,'Vip',616082,6),
                    (1170175289,'Vip',616082,8),
                    (1170365425,'Vip',616082,9),
                    
                    (1169253181,'Vip',616085,1),
                    (1170229215,'Vip',616085,3),
                    (1176593023,'Vip',616085,4),
                    (1173241117,'Vip',616085,5),
                    (1173750915,'Vip',616085,6),
                    (1176214242,'Vip',616085,7),
                    (1174570559,'Vip',616085,8),
                    (1174812495,'Vip',616085,9),
                    
                    (1176560554,'Vip',619408,1),
                    (1176779001,'Vip',619408,2),
                    
                    (1145351299,'Sst',598892,2),
                    
                    #(1161877197,'Sst',613523,1),
                    (1163170091,'Sst',613523,2),
                    (1163554066,'Sst',613523,3),
                    (1164116253,'Sst',613523,4),
                    (1164304690,'Sst',613523,5),
                    #(1164527239,'Sst',613523,6),#NaNs in sweep_table, might not have completed stim
                    (1164751594,'Sst',613523,7),
                    (1164988867,'Sst',613523,8),
                    #(1167079468,'Sst',613523,9),
                    
                    (1173180948,'Sst',615853,1),
                    (1173409438,'Sst',615853,2),
                    (1173698354,'Sst',615853,3),
                    (1173955682,'Sst',615853,4),
                    (1174522035,'Sst',615853,5),
                    (1174750543,'Sst',615853,6),
                    (1175024638,'Sst',615853,7),
                    (1175250452,'Sst',615853,8),
                    #(1175478209,'Sst',615853,9),
                    
                    (1167398732,'Sst',615858,1),
                    (1167912934,'Sst',615858,2),
                    (1168092997,'Sst',615858,3),
                    (1168986071,'Sst',615858,4),
                    (1169218245,'Sst',615858,5),
                    (1175323842,'Sst',615858,6),
                    (1170214686,'Sst',615858,7),
                    (1170409463,'Sst',615858,8),
                    (1172993266,'Sst',615858,9),
                    
                    (1173918056,'Sst',618935,1),
                    (1174492415,'Sst',618935,2),
                    (1175219013,'Sst',618935,3),
                    #(1175447089,'Sst',618935,4),#processing stalled.
                    #(1176085819,'Sst',618935,5),#processing stalled.
                    (1176338780,'Sst',618935,6),
                    (1176522368,'Sst',618935,7),
                    (1176743288,'Sst',618935,8),
                    #(1176945224,'Sst',618935,9),#processing stalled. 
                       
                    ]
    
    session_df = pd.DataFrame(data=np.zeros((len(session_info),4)),
                              columns=('session_ID',
                                       'cre_line',
                                       'mouse_ID',
                                       'Day'))
    
    for i,si in enumerate(session_info):
        session_df.iloc[i] = si
    print(session_df)
    
    return session_df
    
def to_exptpath(datapath,cre,mouse_ID,day):
    exptpath = datapath + cre + '_' + str(mouse_ID) + '/'
    exptpath = exptpath + str(mouse_ID) + '_day' + str(day)+'/'
    return exptpath

if __name__=='__main__':  
    main()
    
    
    
    
    
    
    