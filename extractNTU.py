#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os
import sys
import glob

from utils.AIR import move_camera_to_front, vectorize
from config import path

save_npy_path = path.ntu_test_data
os.makedirs(save_npy_path, exist_ok=True)
load_txt_path = r'D:\HRI DB\NTU Action Recognition\Skeletons\whole'
step_ranges = list(range(0, 100))  # just parse range, for the purpose of paralle running.

toolbar_width = 50


def print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write('\n')


def load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True
    return missing_files


def save_skeleton(load_file_path, save_file_path):
    f = open(load_file_path, 'r')
    datas = f.readlines()
    f.close()

    human_info = list()
    robot_info = list()

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1])
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        if bodycount != 2:
            return

        for body in range(bodycount):
            cursor += 1
            cursor += 1

            njoints = int(datas[cursor][:-1])
            joints = list()
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                joint_xyz = dict()
                joint_xyz['x'] = jointinfo[0]
                joint_xyz['y'] = jointinfo[1]
                joint_xyz['z'] = jointinfo[2]
                joints.append(joint_xyz)

            djoints = dict()
            djoints['joints'] = joints

            if body == 0:
                human_info.append([djoints])
            if body == 1:
                robot_info.append([djoints])

    extracted_human_info = list()
    extracted_robot_info = list()
    extracted_third_info = list()
    human_action_info = list()

    MAX_DISTANCE = 5.
    n_frames = min(len(human_info), len(robot_info))

    for f in range(n_frames):
        dist = np.linalg.norm(vectorize(human_info[f][0]['joints'][0]) - vectorize(robot_info[f][0]['joints'][0]))
        extracted_third_info.append([dist / MAX_DISTANCE])

    # move camera position in front of person
    move_camera_to_front(human_info, body_id=0)
    move_camera_to_front(robot_info, body_id=0)

    for f in range(n_frames):
        extracted_human_info.append(human_info[f][0]["joints"])
        extracted_robot_info.append(robot_info[f][0]["joints"])
        human_action_info.append("stand")

    np.savez(save_file_path,
             human_info=extracted_human_info,
             robot_info=extracted_robot_info,
             third_info=extracted_third_info,
             human_action=human_action_info)

    print(os.path.basename(load_file_path))


if __name__ == '__main__':
    datalist = list()
    for action in ['C001*R001*A058', 'C001*R001*A055', 'C001*R001*A051']:
        datalist.extend(map(os.path.basename, glob.glob(os.path.join(load_txt_path, f"*{action}*.skeleton"))))

    for ind, each in enumerate(datalist):
        print_toolbar(ind * 1.0 / len(datalist), '({:>5}/{:<5})'.format(ind + 1, len(datalist)))
        loadname = os.path.join(load_txt_path, each)
        savename = os.path.join(save_npy_path, os.path.splitext(each)[0])
        if os.path.exists(savename):
            print('file already existed !')
            continue
        S = int(each[1:4])
        if S not in step_ranges:
            continue
        save_skeleton(loadname, savename)
        # raise ValueError()
    end_toolbar()
