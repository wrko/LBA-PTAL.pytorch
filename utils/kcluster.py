import os
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans

from utils.AIR import norm_features
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

# for AIR-Act2Act
sub_action_mapping_1 = {0: 3, 1: 2, 2: 1, 3: 0}
sub_action_mapping_2 = {0: 5, 1: 0, 2: 6, 3: 4}
sub_action_mapping_3 = {0: 6, 1: 8, 2: 9, 3: 10, 4: 7, 5: 0, 6: 10}
sub_action_mapping_4 = {0: 6, 1: 0, 2: 12, 3: 6, 4: 11}
# for NTU data
sub_action_mapping_5 = {0: 0, 1: 4, 2: 0, 3: 4}

subaction_names = list()
subaction_names.append(all_subaction_names[0])
if "A001" in hp.actions:
    subaction_names.append(all_subaction_names[1])
    subaction_names.append(all_subaction_names[2])
    subaction_names.append(all_subaction_names[3])
if "A005" in hp.actions or "A058" in hp.actions:
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
    def __init__(self, actions, n_clusters):
        self.actions = actions
        self.n_clusters = n_clusters

        model_file = os.path.join(path.kmeans_model, f"{''.join(self.actions)}_full_{self.n_clusters}_cluster.pkl")
        if not os.path.exists(model_file):
            print("\nK-means clustering model training...")
            self.train(actions, n_clusters)
        self.km_model = pickle.load(open(model_file, "rb"))

    def make_null_dataframe(self, input_length):
        # feature names
        # p0jB == position of joint B at frame 0
        # vAjB == difference of the joint positions of the frame A and frame A-1
        feature_name = [f"d{f}" for f in range(input_length)]
        feature_name.extend([f"p{f}j{j}" for f in range(input_length) for j in range(pose_dim)])
        feature_name.extend([f"v{f}j{j}" for f in range(1, input_length, 1) for j in range(pose_dim)])
        return pd.DataFrame(columns=feature_name)

    def make_dataframe(self, inputs, input_length):
        df = self.make_null_dataframe(input_length)
        for _, file in enumerate(inputs):  # for each file
            temp = dict()
            for f, data in enumerate(file):  # for each frame
                temp[f"d{f}"] = data[0]
                for j in range(pose_dim):
                    temp[f"p{f}j{j}"] = data[j + 1]
                    if f != 0:
                        temp[f"v{f}j{j}"] = data[j + 1] - temp[f"p{f-1}j{j}"]
            df = df.append(temp, ignore_index=True)
        return df

    def preprocessing(self, input_length, file_names):
        human_data = list()
        third_data = list()
        for file in file_names:
            with np.load(file, allow_pickle=True) as data:
                human_data.append([norm_features(human, norm_method) for human in data['human_info']])
                third_data.append(data['third_info'])

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
        train = self.make_null_dataframe(seq_length)
        for action in actions:
            print(F"\nAction: {action}")
            files = glob.glob(os.path.join(train_path, f"*{action}*.npz"))
            train = train.append(self.preprocessing(seq_length, files), ignore_index=True, sort=False)
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


def load_model(actions, n_clusters):
    kmeans = KMeansClustering(actions=actions, n_clusters=n_clusters)
    return kmeans


def load_proper_model(action):
    if action == 'A001' or action == 'A004':
        kmeans = load_model(["A001", "A004"], 4)
        sub_action_mapping = sub_action_mapping_1
    elif action == 'A005':
        kmeans = load_model(["A004", "A005"], 4)
        sub_action_mapping = sub_action_mapping_2
    elif action == 'A006':
        kmeans = load_model(["A004", "A006"], 7)
        sub_action_mapping = sub_action_mapping_3
    elif action == 'A008':
        kmeans = load_model(["A004", "A008"], 5)
        sub_action_mapping = sub_action_mapping_4
    elif action == 'A058':
        kmeans = load_model(["A058"], 4)
        sub_action_mapping = sub_action_mapping_5
    return kmeans, sub_action_mapping


def test():
    # action list to test
    global train_path
    train_path = path.ntu_extracted
    hp.actions = ["A058"]
    actions = ["A058"]

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
            results = list()
            # action class mapping
            if train_path == path.air_extracted:
                action = os.path.basename(data_file)[4:8]
            if train_path == path.ntu_extracted:
                action = os.path.basename(data_file)[16:20]
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
