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


def get_obj(string):
    SelectionList = OpenMaya.MSelectionList()
    SelectionList.add(string)
    obj = SelectionList.getDagPath(0)
    return obj


def get_mesh_value(obj):
    print(obj)
    MFnMesh = OpenMaya.MFnMesh(obj)
    points = MFnMesh.getPoints(OpenMaya.MSpace.kObject)
    point_list = []
    for i in points:
        point_list.append(i.x)
        point_list.append(i.y)
        point_list.append(i.z)
    return point_list


def save_array(array, save_path):
    array = np.array(array).astype(np.float32)
    return np.save(save_path, array)


def main(controller_file_path, controller_save_path, mesh_node_name, mesh_save_path, frame_num=200):
    controller_name, default_value, min_value, max_value = get_ctrl_name(controller_file_path)

    cmds.currentTime(0)
    for idx, ctrl_value in enumerate(default_value):
        cmds.setKeyframe(controller_name[idx], v=float(ctrl_value), t=0)

    mesh_obj = get_obj(mesh_node_name)
    neutral_head_mesh = np.array(get_mesh_value(mesh_obj))

    for index in range(1, frame_num):
        value = []
        for idx in range(len(controller_name)):
            val = random.random()*(max_value[idx]-min_value[idx])+min_value[idx]
            value.append(val)

        save_array(value, os.path.join(controller_save_path, str(index) + '.npy'))
        cmds.currentTime(index)
        for idx, ctrl_value in enumerate(value):
            cmds.setKeyframe(controller_name[idx], v=float(ctrl_value), t=index)

        point_value = np.array(get_mesh_value(mesh_obj))
        save_array(point_value - neutral_head_mesh, os.path.join(mesh_save_path, str(index) + '.npy'))

    save_array(neutral_head_mesh, os.path.join(mesh_save_path, 'neutralHead.npy'))


if __name__ == "__main__":
    main()