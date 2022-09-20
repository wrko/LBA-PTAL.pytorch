import os
import glob
import math
import numpy as np
from tqdm import tqdm

from utils.draw import Artist
from utils.AIR import norm_features, denorm_features
from config import path, hp


def label(kcluster_module, b_show=False):
    # get all data files
    data_files = list()
    for action in hp.actions:
        data_files.extend(glob.glob(os.path.join(kcluster_module.train_path, F"*{action}*.npz")))
    data_files.sort()

    # label action classes
    pbar = tqdm(total=len(data_files))
    if b_show:
        artist = Artist(n_plot=1)
    for data_file in data_files:
        with np.load(data_file, allow_pickle=True) as data:
            # action class mapping
            if kcluster_module.train_path == path.air_extracted:
                action = os.path.basename(data_file)[4:8]
            if kcluster_module.train_path == path.ntu_extracted:
                action = os.path.basename(data_file)[16:20]
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
                action_name = "None"
                sampled_labels.append(action_name)
                if b_show:
                    features = denorm_features(sampled_human_data[f], kcluster_module.norm_method)
                    frame_info = f"{f+1}/{len(sampled_human_data)}"
                    artist.update('', ['User'], [features], [action_name], [frame_info], fps=10)

            # label recognized action class by k-means clustering
            for human_seq, third_seq in zip(kcluster_module.gen_sequence(sampled_human_data, hp.user_pose_length),
                                            kcluster_module.gen_sequence(sampled_third_data, hp.user_pose_length)):
                seq = np.concatenate((third_seq, human_seq), axis=1)
                df = kmeans.make_dataframe([seq], hp.user_pose_length)
                sub_action = km_model.predict(df)
                action_name = kcluster_module.all_subaction_names[sub_action_mapping[sub_action[0]]]
                sampled_labels.append(action_name)

                if b_show:
                    f += 1
                    features = denorm_features(human_seq[-1], kcluster_module.norm_method)
                    frame_info = f"{f+1}/{len(sampled_human_data)}"
                    artist.update('', ['User'], [features], [action_name], [frame_info], fps=10)

            # print(sampled_labels)

            # add to data
            labels = list()
            for f in range(len(human_data)):
                labels.append(sampled_labels[math.floor(f / step)])

            np.savez(data_file,
                     human_info=data['human_info'],
                     third_info=data['third_info'],
                     human_action=labels)

        pbar.update(1)
    pbar.close()


def label_AIR(b_show=False):
    hp.actions = ['A005']
    kcluster_module = __import__('utils.kcluster', fromlist=['load_proper_model'], level=0)
    kcluster_module.train_path = path.air_extracted
    label(kcluster_module, b_show=b_show)


def label_NTU(b_show=False):
    hp.actions = ['A058']
    kcluster_module = __import__('utils.kcluster', fromlist=['load_proper_model'], level=0)
    kcluster_module.train_path = path.ntu_extracted
    label(kcluster_module, b_show=b_show)


if __name__ == '__main__':
    # label_AIR(b_show=False)
    label_NTU(b_show=False)
