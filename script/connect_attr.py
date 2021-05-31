import maya.cmds as cmds


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


def main(controller_file_path):
    controller_name, default_value, min_value, max_value = get_ctrl_name(controller_file_path)
    for each in controller_name:
        print()
        cmds.connectAttr("%s" % each, "ofCtrlDeformer1.%s" % (each.replace('.', '_')), f=1)


if __name__ == "__main__":
    main()