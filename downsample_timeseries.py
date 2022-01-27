# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 00:46:56 2021

@author: danielm
"""

import h5py
import numpy as np
from skimage.measure import block_reduce

def downsample(timeseries_path,
               downsampled_path,
               frames_to_average = 2,
               batch_size = 10000, #frames
               orig_xdim = 512,
               orig_ydim = 512,
               dataset_fieldname = 'data'
               ):
    
    f = h5py.File(timeseries_path,'r')
    f2 = h5py.File(downsampled_path,'w')
    
    num_frames = f[dataset_fieldname].shape[0]
    if (num_frames % frames_to_average) > 0:
        num_frames -= (num_frames % frames_to_average)
    
    num_batches = int(np.ceil(num_frames / batch_size))
    
    downsampled_frames = np.zeros((int(num_frames/frames_to_average),orig_ydim,orig_xdim),dtype=np.int16)
    
    for i_batch in range(num_batches):
        start_frame = i_batch*batch_size
        end_frame = (i_batch+1)*batch_size
        if end_frame>num_frames:
            end_frame = num_frames
            
        print('batch frames '+str(start_frame)+' to '+str(end_frame))
        batch_frames = f[dataset_fieldname][start_frame:end_frame]
        print('frames loaded')
        batch_downsampled = block_reduce(batch_frames,
                                         block_size=(frames_to_average,1,1),
                                         func=np.mean
                                         )
        print('frames downsampled')
        start_ds = int(start_frame/frames_to_average)
        end_ds = int(end_frame/frames_to_average)
        downsampled_frames[start_ds:end_ds] = batch_downsampled
    
    f.close()
    f2.create_dataset(dataset_fieldname,data=downsampled_frames)
    f2.close()
    
if __name__=='__main__':
    
    timeseries_path = r'E:\\594263_day6\1144953459_timeseries.h5'
    downsampled_path = r'E:\\594263_day6\1144953459_timeseries_ds.h5'
    downsample(timeseries_path,downsampled_path)
    
    timeseries_path = r'E:\\598130_day4\1143779068_timeseries.h5'
    downsampled_path = r'E:\\598130_day4\1143779068_timeseries_ds.h5'
    downsample(timeseries_path,downsampled_path)
    
    timeseries_path = r'E:\\598130_day5\1144578662_timeseries.h5'
    downsampled_path = r'E:\\598130_day5\1144578662_timeseries_ds.h5'
    downsample(timeseries_path,downsampled_path)
    
    timeseries_path = r'E:\\598130_day6\1144819429_timeseries.h5'
    downsampled_path = r'E:\\598130_day6\1144819429_timeseries_ds.h5'
    downsample(timeseries_path,downsampled_path)
    
    timeseries_path = r'E:\\598892_day2\1145351299_timeseries.h5'
    downsampled_path = r'E:\\598892_day2\1145351299_timeseries_ds.h5'
    downsample(timeseries_path,downsampled_path)
    