#!/usr/bin/env python
# coding=utf-8


import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

def create_modules(bl_list):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check type
        # create new module
        # append to module_list

