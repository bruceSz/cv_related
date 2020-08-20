#!/usr/bin/env python
# coding=utf-8


import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(pred, inp_dim, anchors, num_classes, CUDA = False):
    batch_size = pred.size(0)
    stride = inp_dim // pred.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    pred = pred.view(batch_size, bbox_attrs* num_anchors, grid_size * grid_size)
    pred = pred.transpose(1,2).contiguous()
    pred = pred.view(batch_size, grid_size * grid_size * num_anchors,bbox_attrs)
    anchors = [(a(0)/stride, a[1]/stride) for a in anchors]

    pred[:, : , 0] = torch.sigmoid([pred[:,:,0]])
    pred[:, : , 1] = torch.sigmoid([pred[:,:,1]])
    pred[:, : , 4] = torch.sigmoid([pred[:,:,4]])

    # add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_off = torch.FloatTensor(a).view(-1,1)
    y_off = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_off = x_off.cuda()
        y_off = y_off.cuda()

    x_y_off = torch.cat((x_off, y_off),1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    pred[:,:,:2] += x_y_off


    # add anchors to bbox
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    pred[:,:,2:4] = torch.exp(pred[:,:,2:4])*anchors

    # compute class  score
    pred[:,:,5:5+num_classes] = torch.sigmoid(pred[:,:,5:5 + num_classes])


    # resize feature map to input_image size.
    pred[:,:,:4] *= stride

    return pred

    


def parse_cfg(conf):
    """
        Parse the cfg and 
        return list of blocks, which describe a block in nn.
    """
    block = {}
    bl_list = []
    with open(conf, 'r') as f:
        lines = f.read().split('\n')
        lines = [x for x in lines if len(x) > 0]
        lines = [x for x in lines if x[0] != '#']
        lines = [x.rstrip().lstrip() for x in lines]
        
        for l in lines:
            if l[0] == "[":
                if len(block) !=0:
                    # met new block
                    bl_list.append(block)
                    block = {}
                block['type'] = l[1:-1].rstrip()
            else:
                try:
                    k,v = l.split("=")
                except Exception as e:
                    print (l)
                    print(e)
                block[k.rstrip()] = v.lstrip()
        bl_list.append(block)
    return bl_list



def test_cfg():
    p  = "./cfg/yolov3.cfg"

    d = parse_cfg(p)
    for x in d:
        print("block is: ", x)

if __name__ == "__main__":
    test_cfg()