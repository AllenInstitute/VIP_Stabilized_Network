"""
VIPSSN_habituation_day6.py
"""
#from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp
import numpy as np

# Create display window
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0,
                warp=Warp.Disabled
                )

local_path = r'C:\\ProgramData\AIBS_MPE\camstim\resources\VIPSSN\\'
stim_path = local_path + 'VIPSSN_habituation_movie.stim'
stim = Stimulus.from_file(stim_path, window)
    
stim_ds = [(0,3600)]
stim.set_display_sequence(stim_ds)
    
params = {'syncsqr': True,
          #'syncsqrloc': (875,550),
          #'syncsqrsize': (150,150),
          'syncpulse': True,
          'syncpulseport': 1,
          #'syncpulselines': [5, 6],  # frame, start/stop
          'trigger_delay_sec': 5.0}

# create SweepStim instance
ss = SweepStim(window,
               stimuli=[stim],
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
    
    
