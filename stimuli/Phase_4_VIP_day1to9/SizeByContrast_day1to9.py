"""
SizeByContrast_day1to9.py
"""
import sys
from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp
import numpy as np

# optional param file
if len(sys.argv) > 1:
    import json
    with open(sys.argv[1], 'r') as f:
        param_file = json.load(f)
else:
    param_file = {}
posx = param_file.get("posx", 0.0)    #in degrees
posy = param_file.get("posy", 0.0)    #in degrees

# Create display window
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=1,
                warp=Warp.Spherical,
                )

# Paths for stimulus files
size_by_contrast_path = "size_by_contrast.stim"
behavior_flashes_path = "visual_behavior_flashes.stim"

size_by_contrast_sequence_path = "size_by_contrast_sequence.npy"
behavior_flashes_sequence_path = "visual_behavior_flash_sequence.npy"

size_by_contrast_sequence = np.load(size_by_contrast_sequence_path).astype(int)
behavior_flashes_sequence = np.load(behavior_flashes_sequence_path).astype(int)

size_by_contrast = Stimulus.from_file(size_by_contrast_path, window)
behavior_flashes = Stimulus.from_file(behavior_flashes_path, window)

size_by_contrast.stim.pos = (posx, posy)

size_by_contrast.sweep_order = size_by_contrast_sequence[:,0].tolist()
size_by_contrast._build_frame_list()
behavior_flashes.sweep_order = behavior_flashes_sequence.tolist()
behavior_flashes._build_frame_list()

# set display sequences
size_by_contrast_ds=[(0,1725),(2800,4525)]
behavior_flashes_ds=[(1950,2565)]

size_by_contrast.set_display_sequence(size_by_contrast_ds)
behavior_flashes.set_display_sequence(behavior_flashes_ds)

# kwargs
params = {
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [5,6],  # frame, start/stop
    'trigger_delay_sec': 5.0,
}

# create SweepStim instance
ss = SweepStim(window,
               stimuli=[size_by_contrast, behavior_flashes],
               pre_blank_sec=2,
               post_blank_sec=2,
               params=params,
               )

# add in foraging so we can track wheel, potentially give rewards, etc
f = Foraging(window=window,
             auto_update=False,
             params=params,
             nidaq_tasks={'digital_input': ss.di,
                          'digital_output': ss.do,})  #share di and do with SS
ss.add_item(f, "foraging")

# run it
ss.run()
