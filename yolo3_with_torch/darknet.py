#!/usr/bin/env python
# coding=utf-8


import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import utils

import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors


class DarkNet(nn.Module):
    def __init__(self, cfgfile):
        super(DarkNet, self).__init__()
        self.blocks = utils.parse_cfg(cfgfile)
        self.net_info , self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules  = self.blocks[1:]
        outputs = {}
        # cache output here to implement route and shortcut.
        write = 0
        for i , module in enumerate(modules):
            module_t = (module['type'])
            if module_t == "convolutional" or module_t == "upsample":
                x = self.module_list[i](x)
            if module_t == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    # layer from current i.
                    layers[0] = layers[0]-i
                
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]>0):
                        layers[1] = layers[1]  - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_t == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                

                
        

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check type
        # create new module
        # append to module_list
        if x['type'] == "convolutional":
            #Get the info about the layer
            act = x['activation']
            try:
                batch_norm = int(x['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bais = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad  = 0
            #Add the conv layer.
            conv = nn.Conv2d(prev_filters, filters,  kernel_size, 
                stride, pad, bias = bias)

            module.add_module("conv_{0}".format(index),conv)

            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)

            # check act
            if act == "leaky":
                actvn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), actvn)
            
        elif (x['type'] == "upsample"):
            stride  = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        elif (x['type'] == "route"):
            x['layers'] = x['layers'].split(",")
            start = int(x['layers'][0])

            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - index
                
            if end > 0:
                end = end - index
            
            route  = EmptyLayer()

            module.add_module("route_{0}".format(index),route)
            
            if end < 0:
                filters = output_filters[index + start]  \
                      + output_filters[index + end]
            else:
                filters  = output_filters[index + start]
        elif (x['type'] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index),shortcut)

        elif (x['type'] == "yolo") :
            mask = x['mask'].split(",")
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i] , anchors[i+1]) for i in range(0,len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)





def test():
    blocks = utils.parse_cfg("cfg/yolov3.cfg")
    print(create_modules(blocks))


if __name__ == "__main__":
    test()