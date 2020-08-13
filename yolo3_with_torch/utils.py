#!/usr/bin/env python
# coding=utf-8


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