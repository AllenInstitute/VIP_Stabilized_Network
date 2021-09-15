#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:02:34 2021

@author: danielm
"""
import sys
import day0_sparse_noise_analysis

mouse_ID = sys.argv[1]
session_ID = sys.argv[2]
session_type = sys.argv[3]

exptpath = r'S:\\scientifica_ophys\\data\\' + str(session_ID) + '\\'

is_bessel = False
if session_type.lower().find('bessel'):
    is_bessel = True

day0_sparse_noise_analysis.run_analysis(mouse_ID=mouse_ID,exptpath=exptpath,is_bessel=is_bessel)