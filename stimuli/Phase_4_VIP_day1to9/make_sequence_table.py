#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:00:33 2021

@author: danielm
"""

import numpy as np

def make_SizeByContrast_sequence_table(num_reps = 23,
                        savepath = '/Users/danielm/Desktop/stimuli/'
                        ):
    
    TFs = [1]
    SFs = [0.08]
    Sizes = [12.5,25,250]
    Contrasts = [2.5,5,10,20,40,80]
    Oris = [0,90,180,270]
    
    #params = [TFs,SFs,Sizes,Contrasts,Oris]
    
    num_conditions = int(len(Oris)*(len(Contrasts)+3*(len(Sizes)-1)) + 2)
    
    one_rep = np.zeros((num_conditions,)).astype(np.int)
    
    #6 contrasts at full-field
    one_rep[:24] = np.arange(48,72)
    
    #small size at 5, 20, 80 contrast
    one_rep[24:36] = np.array([4,5,6,7,12,13,14,15,20,21,22,23]).astype(np.int)
    
    #intermediate size at 5, 20, 80 contrast
    one_rep[36:48] = np.array([4,5,6,7,12,13,14,15,20,21,22,23]).astype(np.int) + 24
    
    #blank sweeps, twice as many as other conditions (50)
    one_rep[-1] = -1
    one_rep[-2] = -1

    #make full sequence by shuffling repeats
    full_sequence = np.zeros((num_conditions*num_reps,1)).astype(np.int)
    for nr in range(num_reps):
        full_sequence[(nr*num_conditions):((nr+1)*num_conditions),0] = np.random.permutation(one_rep)

    np.save(savepath+'size_by_contrast_sequence.npy',full_sequence)

def make_behavior_flash_sequence_table(min_flashes = 3,
                                       max_flashes = 9,
                                       num_images = 5,
                                       num_reps = 9,
                                       prob_per_flash = 0.4,
                                       omission_rate = 0.05,
                                       savepath = '/Users/danielm/Desktop/stimuli/'
                                       ):
    
    num_transitions = num_images * num_images
    num_trials = num_transitions * num_reps
    
    #determine order of images
    image_sequence = np.zeros((num_trials,)).astype(np.uint8)
    for rep in range(num_reps):
        one_rep = None
        while one_rep is None:
            one_rep = try_one_behavior_rep(num_images)
        image_sequence[(rep*num_transitions):((rep+1)*num_transitions)] = one_rep
    print(image_sequence)
    
    #compute distribution of flashes per trial
    trial_time = 0.75 * (1+np.arange(max_flashes,))
    reps_per_image = num_images * num_reps
    prob_of_N_flashes = np.zeros((max_flashes,))
    for nf in range(min_flashes,max_flashes):
        prob_of_N_flashes[nf] = prob_per_flash ** (nf - min_flashes + 1)
    prob_of_N_flashes = prob_of_N_flashes / np.sum(prob_of_N_flashes)
    num_trials_with_N_flashes = np.round(prob_of_N_flashes*reps_per_image)
    num_trials_with_N_flashes[6] = 2
    mean_trial_time = (num_trials_with_N_flashes * trial_time).sum()/np.sum(num_trials_with_N_flashes)
    print(trial_time)
    print(prob_of_N_flashes)
    print(mean_trial_time)
    print(num_trials_with_N_flashes)
    
    #assign flash number to trials for each image
    flashes_per_trial = np.zeros((num_trials,)).astype(np.uint8)
    for i_image in range(num_images):
        
        im_flashes_per_trial = []
        for nf in range(max_flashes):
            for nt in range(int(num_trials_with_N_flashes[nf])):
                im_flashes_per_trial.append(nf)
        im_flashes_per_trial = np.array(im_flashes_per_trial)
        
        trials_this_image = np.argwhere(image_sequence==i_image)[:,0]
        flashes_per_trial[trials_this_image] = np.random.permutation(im_flashes_per_trial)
    print(flashes_per_trial)
    
    #make full presentation sequence
    full_sequence = []
    is_change = []
    for i_trial in range(len(flashes_per_trial)):
        for i_flash in range(flashes_per_trial[i_trial]):
            full_sequence.append(image_sequence[i_trial])
            if i_flash==0:
                is_change.append(True)
            else:
                is_change.append(False)
    full_sequence = np.array(full_sequence).astype(np.int)
    is_change = np.array(is_change).astype(np.bool)
    
    #add omissions
    num_non_change_flashes = int(np.sum(~is_change))
    num_omissions = int(omission_rate*num_non_change_flashes)
    non_change_flash_idx = np.argwhere(~is_change)[:,0]
    omission_flash_idx = np.random.permutation(non_change_flash_idx)[:num_omissions]
    full_sequence[omission_flash_idx] = -1
    
    print(full_sequence)
    print('omissions: '+str(num_omissions))
    print(0.75*len(full_sequence))
    
    np.save(savepath+'visual_behavior_flash_sequence.npy',full_sequence)

def try_one_behavior_rep(num_images):
    
    # one rep goes through all pairwise image transitions:
    num_transitions = num_images * num_images
    available_transitions = np.ones((num_images,num_images)).astype(np.bool)
    
    image_order = np.zeros((num_transitions,)).astype(np.int)
    image_order[0] = np.random.choice(num_images)
    available_transitions[np.random.choice(num_images),image_order[0]] = False #remove one way of transitioning to the seed image
    for i_trial in range(num_transitions-1):
        curr_image = image_order[i_trial]
        possible_next_images = np.argwhere(available_transitions[curr_image])[:,0]
        if len(possible_next_images)==0:
            print(available_transitions.sum())
            return None
        next_image = possible_next_images[np.random.choice(len(possible_next_images))]
        available_transitions[curr_image,next_image] = False
        image_order[i_trial+1] = next_image
        
    print('Found! '+str(available_transitions.sum()))
        
    return image_order
    

if __name__ == '__main__':
    make_SizeByContrast_sequence_table()
    make_behavior_flash_sequence_table()