#-*- coding: utf-8 -*-
"""
Created on 2020. 12. 03. (Thur) 12:54:43 KST

@author: youngjae-park
"""

import os
import h5py

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        print("mkdir error")

def load_hdf5(path, using='Train'):
    dat = h5py.File(path, 'r')
    coord, laby, matx, maty = dat['coord'][()], dat['laby'][()], dat['matx'][()], dat['maty'][()]
    
    if using == 'Train' or using == 'Validation':
        return coord, matx, laby
    elif using == 'Test':
        return coord, matx

