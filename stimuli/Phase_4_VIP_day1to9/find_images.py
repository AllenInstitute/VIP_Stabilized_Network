#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:39:03 2021

@author: danielm
"""
import numpy as np
import matplotlib.pyplot as plt

savepath = '/Users/danielm/Desktop/stimuli/'

natural_scenes = np.load(savepath+'natural_scenes.npy')

print(np.shape(natural_scenes))

images_to_use = [17,25,38,39,42]

for i_scene in images_to_use:
    
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(natural_scenes[:,:,i_scene],cmap='binary_r',vmin=0,vmax=255,interpolation='none',origin='lower')
    ax.set_title(str(i_scene))
    ax.set_aspect('auto')
    plt.show()
    
ns_set = natural_scenes[:,:,images_to_use]
print(np.shape(ns_set))

np.save(savepath+'visual_behavior_images.npy',ns_set)