import os
import glob
import random
import shutil
import numpy as np
from tqdm import tqdm

from utils.AIR import read_joint, vectorize, move_camera_to_front, norm_features
from config import hp, path

# parameters
PROB_TRAIN = .9
DEVIDE = 'scene'  # {scene, subject}
MAX_DISTANCE = 5.  # maximum distance between camera and human
B_OVERWRITE = False

JOINT_PATH = path.air_data
DATA_PATH = path.air_extracted
TRAIN_PATH = path.air_train_data
TEST_PATH = path.air_test_data


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


def split_train_valid():
    reset_train = glob.glob(os.path.join(TRAIN_PATH, "*.npz"))
    for file in reset_train:
        shutil.move(file, os.path.join(DATA_PATH, os.path.basename(file)))
    reset_valid = glob.glob(os.path.join(TEST_PATH, "*.npz"))
    for file in reset_valid:
        shutil.move(file, os.path.join(DATA_PATH, os.path.basename(file)))

    action_names = list()
    files = glob.glob(os.path.join(DATA_PATH, "*.npz"))
    for file in files:
        file_name = os.path.basename(file)
        action_name = file_name[4:8]
        if action_name not in action_names:
            action_names.append(action_name)

    train = list()
    for action_name in action_names:
        data_names = list()
        action_files = glob.glob(os.path.join(DATA_PATH, f"*{action_name}*.npz"))
        for action_file in action_files:
            file_name = os.path.basename(action_file)
            data_name = file_name[:4] if DEVIDE == 'subject' else file_name[:8]
            if data_name not in data_names:
                data_names.append(data_name)
        random.shuffle(data_names)
        train.extend(data_names[:int(len(data_names)*PROB_TRAIN)])

    for file in files:
        file_name = os.path.basename(file)
        data_name = file_name[:4] if DEVIDE == 'subject' else file_name[:8]
        if data_name in train:
            shutil.move(file, os.path.join(TRAIN_PATH, file_name))
        else:
            shutil.move(file, os.path.join(TEST_PATH, file_name))


def label():
    kcluster_module = __import__('utils.kcluster', fromlist=['BodyGameRuntime'], level=0)

    # get all data files
    data_files = list()
    for action in hp.actions:
        data_files.extend(glob.glob(os.path.join(TEST_PATH, F"*{action}*.npz")))
        data_files.extend(glob.glob(os.path.join(TRAIN_PATH, F"*{action}*.npz")))
    data_files.sort()

    # label action classes
    pbar = tqdm(total=len(data_files))
    for data_file in data_files:
        with np.load(data_file, allow_pickle=True) as data:
            # action class mapping
            action = os.path.basename(data_file)[4:8]
            kmeans, sub_action_mapping = kcluster_module.load_proper_model(action)
            km_model = kmeans.km_model

            # extract inputs from data file
            human_data = [norm_features(human, method=hp.input_norm_method) for human in data['human_info']]
            third_data = data['third_info']

            step = hp.step
            sampled_human_data = human_data[::step]
            sampled_third_data = third_data[::step]

            # label "None"
            sampled_labels = list()
            for f in range(hp.user_pose_length - 1):
                sampled_labels.append("None")

            # label recognized action class by k-means clustering
            for human_seq, third_seq in zip(kcluster_module.gen_sequence(sampled_human_data, hp.user_pose_length),
                                            kcluster_module.gen_sequence(sampled_third_data, hp.user_pose_length)):
                seq = np.concatenate((third_seq, human_seq), axis=1)
                df = kmeans.make_dataframe([seq], hp.user_pose_length)
                sub_action = km_model.predict(df)
                action_name = kcluster_module.subaction_names[sub_action_mapping[sub_action[0]]]
                sampled_labels.append(action_name)

            # add to data
            labels = list()
            for f in range(len(human_data)):
                labels.append(sampled_labels[int(f / step)])

            np.savez(data_file,
                     human_info=data['human_info'],
                     third_info=data['third_info'],
                     human_action=labels)

        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

    # generate data files
    gen_datafiles()

    # split data into train and validate sets
    split_train_valid()

    # label sub-action classes
    label()
