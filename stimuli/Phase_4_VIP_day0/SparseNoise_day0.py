"""
SparseNoise_day0.py
"""
from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp
import numpy as np

# Create display window
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=1,#TODO: verify screen number for DeepScope
                warp=Warp.Spherical,
                )

# Paths for stimulus files
sn_path  = "sparse_noise_9on4.stim"

# Create stimulus
sn = Stimulus.from_file(sn_path, window)

# set display sequences
sn_ds=[(0,1519),(1819,3338)]
sn.set_display_sequence(sn_ds)

# kwargs
params = {
    'syncsqrloc': (510,360),#TODO: verify for DeepScope
    'syncsqrsize': (50,140),#TODO: verify for DeepScope
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [1,2],#TODO: verify for DeepScope
    'trigger_delay_sec': 5.0,
}

# create SweepStim instance
ss = SweepStim(window,
               stimuli=[sn],
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
