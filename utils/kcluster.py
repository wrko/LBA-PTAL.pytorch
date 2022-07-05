import os
import glob
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans

from utils.AIR import norm_features, denorm_features
from utils.draw import Artist
from config import path, hp

# path
train_path = path.air_extracted

# parameters
seq_length = 15
pose_dim = 3 * 8
norm_method = 'vector'
step = 3

# sub-actions
all_subaction_names = dict()
all_subaction_names[0] = "stand"
all_subaction_names[1] = "open the door"
all_subaction_names[2] = "hand on wall"
all_subaction_names[3] = "not shown"
all_subaction_names[4] = "raise right hand"
all_subaction_names[5] = "wave right hand"
all_subaction_names[6] = "lower hands"
all_subaction_names[7] = "cry with right hand"
all_subaction_names[8] = "cry with left hand"
all_subaction_names[9] = "raise both hands"
all_subaction_names[10] = "cry with both hands"
all_subaction_names[11] = "threaten to hit with right hand"
all_subaction_names[12] = "threaten to hit with left hand"

sub_action_mapping_1 = {0: 0, 1: 3, 2: 2, 3: 1}
sub_action_mapping_2 = {0: 0, 1: 5, 2: 4, 3: 6}
sub_action_mapping_3 = {0: 6, 1: 0, 2: 9, 3: 8, 4: 7, 5: 10}
sub_action_mapping_4 = {0: 6, 1: 6, 2: 0, 3: 12, 4: 11}

subaction_names = list()
subaction_names.append(all_subaction_names[0])
if "A001" in hp.actions:
    subaction_names.append(all_subaction_names[1])
    subaction_names.append(all_subaction_names[2])
    subaction_names.append(all_subaction_names[3])
if "A005" in hp.actions:
    subaction_names.append(all_subaction_names[4])
    subaction_names.append(all_subaction_names[5])
if "A005" in hp.actions or "A006" in hp.actions or "A008" in hp.actions:
    subaction_names.append(all_subaction_names[6])
if "A006" in hp.actions:
    subaction_names.append(all_subaction_names[7])
    subaction_names.append(all_subaction_names[8])
    subaction_names.append(all_subaction_names[9])
    subaction_names.append(all_subaction_names[10])
if "A008" in hp.actions:
    subaction_names.append(all_subaction_names[11])
    subaction_names.append(all_subaction_names[12])


# function to extract sequence from data
def gen_sequence(data, length):
    for start_idx in range(len(data) - length + 1):
        yield list(data[start_idx:start_idx + length])


class KMeansClustering:
    def __init__(self, actions, n_clusters, seq_length, norm_method):
        self.actions = actions
        self.n_clusters = n_clusters
        self.seq_length = seq_length
        self.norm_method = norm_method

        model_file = os.path.join(path.kmeans_model, f"{''.join(self.actions)}_full_{self.n_clusters}_cluster.pkl")
        if not os.path.exists(model_file):
            print("K-means clustering model training...")
            self.train(actions, n_clusters)
        self.km_model = pickle.load(open(model_file, "rb"))

    def make_null_dataframe(self, input_length):
        # feature names
        # fAd == distance between human and robot at frame A
        # fAjB == position of joint B at frame A
        feature_name = [F"f{a}d" for a in range(input_length)]  # f0d ~ f14d
        feature_name.extend(
            [F"f{a}j{b+1}" for a in range(input_length) for b in range(pose_dim)])  # f0j1 ~ f0j30, f1j1 ~ f1j30, ...

        return pd.DataFrame(columns=feature_name)

    def make_dataframe(self, inputs, input_length):
        df = self.make_null_dataframe(input_length)
        for i, a in enumerate(inputs):  # for each file
            temp = dict()
            for j, b in enumerate(a):  # for each sequence of 15 frames
                for k, c in enumerate(b):  # for each frame
                    if k == 0:
                        temp[F"f{j}d"] = c  # distance
                    else:
                        temp[F"f{j}j{k}"] = c
            df = df.append(temp, ignore_index=True)

        # additional features (vj0: difference between the joint positions of the first and last frames
        for i in range(pose_dim):
            df[F"vj{i}"] = df[F"f0j{i+1}"] - df[F"f{input_length-1}j{i+1}"]

        return df

    def preprocessing(self, input_length, file_names):
        human_data = list()
        third_data = list()

        pbar = tqdm(total=len(file_names))
        for file in file_names:
            with np.load(file, allow_pickle=True) as data:
                human_data.append([norm_features(human, self.norm_method) for human in data['human_info']])
                third_data.append(data['third_info'])
            pbar.update(1)
        pbar.close()

        inputs = list()
        for idx, third in enumerate(third_data):
            if all(v == 1.0 for v in third):
                continue
            sampled_human_seq = human_data[idx][::step]
            sampled_third_seq = third_data[idx][::step]
            for human_seq, third_seq in zip(gen_sequence(sampled_human_seq, input_length),
                                            gen_sequence(sampled_third_seq, input_length)):
                inputs.append(np.concatenate((third_seq, human_seq), axis=1))

        return self.make_dataframe(inputs, input_length)

    def train(self, actions, n_clusters):
        # gather all data in train_path
        train = self.make_null_dataframe(self.seq_length)
        for action in actions:
            print(F"\nAction: {action}")
            files = glob.glob(os.path.join(train_path, f"*{action}*.npz"))
            train = train.append(self.preprocessing(self.seq_length, files), ignore_index=True, sort=False)
            print(f'Data loaded. Total {len(files)} files.')
        print(f'Total data size: {train.size}')

        # K-means clustering
        print('\nK-means clustering...')
        km = KMeans(n_clusters=n_clusters, random_state=2020)
        km.fit(train)

        # save model
        if not os.path.exists(path.kmeans_model):
            os.makedirs(path.kmeans_model)
        with open(f"{path.kmeans_model}/{''.join(self.actions)}_full_{self.n_clusters}_cluster.pkl", "wb") as f:
            pickle.dump(km, f)
        print('Model saved.')


def load_model(actions, n_clusters, seq_length, norm_method):
    kmeans = KMeansClustering(actions=actions, n_clusters=n_clusters, seq_length=seq_length, norm_method=norm_method)
    return kmeans


def load_proper_model(action):
    if action == 'A001' or action == 'A004':
        kmeans = load_model(["A001", "A004"], 4, seq_length, norm_method)
        sub_action_mapping = sub_action_mapping_1
    elif action == 'A005':
        kmeans = load_model(["A004", "A005"], 4, seq_length, norm_method)
        sub_action_mapping = sub_action_mapping_2
    elif action == 'A006':
        kmeans = load_model(["A004", "A006"], 6, seq_length, norm_method)
        sub_action_mapping = sub_action_mapping_3
    elif action == 'A008':
        kmeans = load_model(["A004", "A008"], 5, seq_length, norm_method)
        sub_action_mapping = sub_action_mapping_4
    return kmeans, sub_action_mapping


def test():
    # action list to test
    actions = ["A008"]

    # show all test data
    data_files = list()
    for action in actions:
        data_files.extend(glob.glob(os.path.join(train_path, F"*{action}*.npz")))
    data_files.sort()
    n_data = len(data_files)

    print('There are %d data.' % n_data)
    for data_idx in range(n_data):
        print('%d: %s' % (data_idx, os.path.basename(data_files[data_idx])))

    # select data name to draw
    artist = Artist(n_plot=1)
    while True:
        var = int(input("Input data number to display: "))
        data_file = data_files[var]

        with np.load(data_file, allow_pickle=True) as data:
            # action class mapping
            print(os.path.basename(data_file))
            action = os.path.basename(data_file)[4:8]
            kmeans, sub_action_mapping = load_proper_model(action)
            km_model = kmeans.km_model

            # extract inputs from data file
            human_data = [norm_features(human, norm_method) for human in data['human_info']]
            third_data = data['third_info']

            # draw data from start start
            sampled_human_data = human_data[::3]
            sampled_third_data = third_data[::3]
            for f in range(seq_length - 1):
                features = denorm_features(sampled_human_data[f], norm_method)
                action_info = "None"
                frame_info = f"{f+1}/{len(sampled_human_data)}"
                artist.update('', ['User'], [features], [action_info], [frame_info], fps=10)

            # recognize sub-action
            for human_seq, third_seq in zip(gen_sequence(sampled_human_data, seq_length),
                                            gen_sequence(sampled_third_data, seq_length)):
                seq = np.concatenate((third_seq, human_seq), axis=1)
                df = kmeans.make_dataframe([seq], seq_length)
                sub_action = km_model.predict(df)
                action_name = all_subaction_names[sub_action_mapping[sub_action[0]]]
                action_name_with_number = f'{sub_action[0]}: {action_name}'
                print(action_name_with_number)

                f += 1
                features = denorm_features(human_seq[-1], norm_method)
                frame_info = f"{f+1}/{len(sampled_human_data)}"
                artist.update('', ['User'], [features], [action_name_with_number], [frame_info], fps=10)


def test_all():
    # action list to test
    actions = ["A005"]

    # show all test data
    data_files = list()
    for action in actions:
        data_files.extend(glob.glob(os.path.join(train_path, F"*{action}*.npz")))
    data_files.sort()
    n_data = len(data_files)
    print('There are %d data.' % n_data)

    # test each data
    for data_file in data_files:
        with np.load(data_file, allow_pickle=True) as data:
            # initialize
            print(os.path.basename(data_file))
            results = list()
            # action class mapping
            action = os.path.basename(data_file)[4:8]
            kmeans, sub_action_mapping = load_proper_model(action)
            km_model = kmeans.km_model
            # extract inputs from data file
            human_data = [norm_features(human, norm_method) for human in data['human_info']]
            third_data = data['third_info']
            # recognize sub-action
            sampled_human_data = human_data[::3]
            sampled_third_data = third_data[::3]
            for human_seq, third_seq in zip(gen_sequence(sampled_human_data, seq_length),
                                            gen_sequence(sampled_third_data, seq_length)):
                seq = np.concatenate((third_seq, human_seq), axis=1)
                df = kmeans.make_dataframe([seq], seq_length)
                sub_action = km_model.predict(df)
                results.append(sub_action_mapping[sub_action[0]])
            # print results
            print(os.path.basename(data_file))
            print(results)


if "__main__" == __name__:
    test()
