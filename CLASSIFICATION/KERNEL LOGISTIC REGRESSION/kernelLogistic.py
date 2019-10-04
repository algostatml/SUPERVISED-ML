#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:49:47 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss
from Utils.kernels import Kernels

class klogistic(EvalC, loss, Kernels):
    def __init__(self):
        return
    
    def fit(self, X, y):
        return
    
    def predict(self, X):
        return
    
#%% Test