import os
import glob
import random
import numpy as np
from tqdm import tqdm

from utils.AIR import read_joint, vectorize, move_camera_to_front, norm_features
from config import hp, path

# parameters
MAX_DISTANCE = 5.  # maximum distance between camera and human
B_OVERWRITE = False

JOINT_PATH = path.air_data
DATA_PATH = path.air_extracted


def gen_datafiles():
    human_files = list()
    for action in hp.actions:
        human_files.extend(glob.glob(os.path.normpath(os.path.join(JOINT_PATH, f"C001*{action}*.joint"))))
    human_files.sort()
    n_data = len(human_files)

    pbar = tqdm(total=n_data)
    for human_file in human_files:
        # skip if .npz file already exists
        data_name = human_file.replace("\\", "/").split("/")[-1].split('.')[0]
        data_file = os.path.join(DATA_PATH, f"{data_name[4:]}.npz")
        if os.path.exists(data_file) and not B_OVERWRITE:
            pbar.update(1)
            continue

        # skip if robot and third view files not exist
        third_file = human_file.replace('C001', 'C003')
        if not os.path.exists(third_file):
            continue

        # read files
        human_info = read_joint(human_file)
        third_info = read_joint(third_file)

        extracted_human_info = list()
        extracted_third_info = list()

        # extract distance features first
        n_frames = min(len(human_info), len(third_info))
        for f in range(n_frames):
            n_body = sum(1 for b in third_info[f] if b is not None)
            if n_body != 2:
                if 'A001' not in human_file and 'A010' not in human_file:
                    raise Exception(f'third camera information is wrong. ({third_file})')
                extracted_third_info.append([MAX_DISTANCE / MAX_DISTANCE])
                continue

            robot_pos1 = vectorize(third_info[f][0]["joints"][0])
            human_pos1 = vectorize(third_info[f][1]["joints"][0])
            dist_third = MAX_DISTANCE if all(v == 0 for v in human_pos1) else np.linalg.norm(human_pos1 - robot_pos1)

            dist_human = MAX_DISTANCE
            if human_info[f][1] is not None:
                human_pos2 = vectorize(human_info[f][1]["joints"][0])
                robot_pos2 = np.array([0., 0., 0.])
                dist_human = MAX_DISTANCE if all(v == 0 for v in human_pos2) else np.linalg.norm(human_pos2 - robot_pos2)

            dist = min(dist_third, dist_human)
            extracted_third_info.append([dist / MAX_DISTANCE])

        # move camera position in front of person
        move_camera_to_front(human_info, body_id=1)

        for f in range(n_frames):
            extracted_human_info.append(human_info[f][1]["joints"])

        np.savez(data_file,
                 human_info=extracted_human_info,
                 third_info=extracted_third_info)

        pbar.update(1)

    pbar.close()


def split_files(files, ratio, devide):
    # list up the subject, action, scene, and data names
    subject_names, action_names, scene_names, data_names = list(), list(), list(), list()
    for file in files:
        file_name = os.path.basename(file)
        if file_name[:4] not in subject_names:
            subject_names.append(file_name[:4])
        if file_name[4:8] not in action_names:
            action_names.append(file_name[4:8])
        if file_name[8:12] not in scene_names:
            scene_names.append(file_name[8:12])
        if file_name[:-4] not in data_names:
            data_names.append(file_name[:-4])

    # names to devide
    if devide == 'random':
        names_to_devide = data_names
    elif devide == 'scene':
        names_to_devide = scene_names
    elif devide == 'subject':
        names_to_devide = subject_names
    else:
        raise Exception(f'Devision criterion is wribg. ({devide})')
    random.shuffle(names_to_devide)
    names_for_train = names_to_devide[:int(len(names_to_devide)*ratio)]

    # split data
    train_files = list()
    test_files = list()
    for action_name in action_names:
        action_files = [file for file in files if action_name in file]
        for action_file in action_files:
            if any(name in action_file for name in names_for_train):
                train_files.append(action_file)
            else:
                test_files.append(action_file)
    return train_files, test_files


if __name__ == "__main__":
    # generate data files
    print('Extracting data...')
    gen_datafiles()

    # # split data into train, retrain and test sets
    # print('Splitting data into train, re-train, test sets...')
    # files = glob.glob(os.path.join(DATA_PATH, "*.npz"))
    # train_files, rest_files = split_files(files, 49 / 50, 'subject')
    # retrain_files, test_files = split_files(rest_files, 1 / 2, 'random')
    # print(len(train_files), len(retrain_files), len(test_files))
