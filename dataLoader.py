# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from tqdm import tqdm

from torch.utils import data as dt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.AIR import norm_features
from utils.kcluster import subaction_names
from config import hp


class HARDataSet(dt.Dataset):
    def __init__(self, data_path, actions, data_name=None, b_pbar=True):
        # load data from files
        self.data_path = data_path
        self.data_name = data_name
        self.actions = actions
        self.file_names = list()
        if data_name is not None:
            self.file_names.extend(glob.glob(os.path.join(self.data_path, self.data_name)))
        else:
            for action in self.actions:
                self.file_names.extend(glob.glob(os.path.join(self.data_path, f"*{action}*.npz")))
        path = data_path if data_name is None else os.path.join(data_path, data_name)
        if b_pbar:
            print(f'Data loading... ({path})')
            print(f'Total {len(self.file_names)} files.')

        self.human_data = list()
        self.third_data = list()
        self.human_actions = list()
        step = hp.step
        for file in self.file_names:
            with np.load(file, allow_pickle=True) as data:
                self.human_data.append([norm_features(human, method='vector') for human in data['human_info']][::step])
                self.third_data.append(data['third_info'][::step])
                self.human_actions.append(data['human_action'][::step])

        # extract training data
        self.inps = list()
        self.outs = list()
        pbar = tqdm(total=len(self.third_data)) if b_pbar else None
        for idx, third in enumerate(self.third_data):
            if all(v == 1.0 for v in third):
                continue
            for human_past_data, human_action in self.extract_data(
                    self.human_data[idx], self.human_actions[idx], self.third_data[idx], hp.user_pose_length):
                self.inps.append(human_past_data)
                cur_action = subaction_names.index(human_action)
                self.outs.append(cur_action)
            for f in range(hp.hold_last):
                self.inps.append(self.inps[-1])
                self.outs.append(self.outs[-1])
            pbar.update(1) if b_pbar else None
        pbar.close() if b_pbar else None

        print(f'Total {len(self.inps)} sequences.') if b_pbar else None

    def __len__(self):
        return len(self.inps)

    def __getitem__(self, item):
        if hp.b_use_noise:
            noise_mean = 0
            noise_var = 0.01
        else:
            noise_mean = 0
            noise_var = 0
        inp = self.inps[item] + np.random.normal(noise_mean, noise_var, self.inps[item].shape)
        return inp.astype("float32"), self.outs[item]

    @staticmethod
    def extract_data(human_data, human_actions, third_data, seq_length):
        for start_idx in range(len(human_data) - seq_length + 1):
            human_past_poses = human_data[start_idx:start_idx + seq_length]
            human_past_dists = third_data[start_idx:start_idx + seq_length]
            human_past_data = np.concatenate((human_past_poses, human_past_dists), axis=1)
            yield np.array(human_past_data), human_actions[start_idx + seq_length - 1]

    def augment_data(self, ):
        # TODO: add data augmentation
        return
