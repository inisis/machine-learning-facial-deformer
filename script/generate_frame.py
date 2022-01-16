import os
import random
import numpy as np
import maya.cmds as cmds
import maya.api.OpenMaya as OpenMaya


def get_ctrl_name(controller_file_name):
    plug_name_list = []
    default_value = []
    min_value = []
    max_value = []

    with open(controller_file_name) as f:
        for line in f:
            line = line.strip('\n')
            if line == '':
                continue                
            names = line.split(' ')
            coords = ['x', 'y', 'z']
            if names[2] in coords:
                name = names[0] + '.' + names[1] + names[2]
            else:
                name = names[0] + '.' + names[2]
            
            min_value.append(float(names[3]))
            max_value.append(float(names[4]))
            
            if(len(names)) < 6:
                default = 0
                default_value.append(default)
            else:
                default = float(names[5])
                default_value.append(default)

            plug_name_list.append(name)

    return plug_name_list, default_value, min_value, max_value


def main(controller_file_path, frame_num=200):
    controller_name, default_value, min_value, max_value = get_ctrl_name(controller_file_path)
    cmds.currentTime(0)
    for idx, ctrl_value in enumerate(default_value):
        cmds.setKeyframe(controller_name[idx], v=float(ctrl_value), t=0)

    for index in range(1, frame_num):
        value = []
        for idx in range(len(controller_name)):
            val = random.random()*(max_value[idx]-min_value[idx])+min_value[idx]
            value.append(val)

        cmds.currentTime(index)
        for idx, ctrl_value in enumerate(value):
            cmds.setKeyframe(controller_name[idx], v=float(ctrl_value), t=index)


if __name__ == "__main__":
    main()
