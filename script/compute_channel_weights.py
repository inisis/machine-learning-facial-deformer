import os
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

            if (len(names)) < 6:
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
    return np.array(point_list).astype(np.float32)


def main(controller_file_path=r"D:\data\ray\ctrlName.txt", mesh_node_name="head_mdl", save_path="./"):
    controller_name, default_value, min_value, max_value = get_ctrl_name(controller_file_path)
    mesh_obj = get_obj(mesh_node_name)

    pose = np.asarray(default_value)
    cmds.currentTime(0)
    for idx, ctrl_value in enumerate(pose):
        cmds.setKeyframe(controller_name[idx], v=float(ctrl_value), t=0)

    total_index = 1

    for index in range(len(controller_name)):
        new_pose = pose.copy()
        new_pose[index] = min_value[index]
        cmds.currentTime(total_index)
        for idx in range(len(new_pose)):
            cmds.setKeyframe(controller_name[idx], v=float(new_pose[idx]), t=total_index)
        total_index += 1

        new_pose = pose.copy()
        new_pose[index] = max_value[index]
        cmds.currentTime(total_index)
        for idx in range(len(new_pose)):
            cmds.setKeyframe(controller_name[idx], v=float(new_pose[idx]), t=total_index)
        total_index += 1

    # should split data generation and save, or there are points offsets difference in Maya mel script
    cmds.currentTime(0)

    neutral_head_mesh = get_mesh_value(mesh_obj)
    neutral_head_mesh = neutral_head_mesh.reshape(-1, 3)

    mask = np.zeros((len(controller_name), len(neutral_head_mesh)))

    total_index = 1
    eps = 1e-4

    for index in range(len(controller_name)):
        cmds.currentTime(total_index)

        head = get_mesh_value(mesh_obj)
        head = head.reshape(-1, 3)
        dist = np.linalg.norm(head - neutral_head_mesh, axis=1)
        mask[index, dist > eps] = 1
        total_index += 1

        cmds.currentTime(total_index)
        head = get_mesh_value(mesh_obj)
        head = head.reshape(-1, 3)
        dist = np.linalg.norm(head - neutral_head_mesh, axis=1)
        mask[index, dist > eps] = 1
        total_index += 1

    np.save(os.path.join(save_path, 'weight_map.npy'), mask)


if __name__ == "__main__":
    main()