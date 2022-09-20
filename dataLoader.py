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
    def __init__(self, data_path, actions=None, data_name=None, data_files=None, b_pbar=True):
        # initialize
        self.data_path = data_path
        self.b_pbar = b_pbar

        # load files
        if [actions, data_name, data_files].count(None) != 2:
            raise Exception('Only one argument must be given among actions, data_name, and data_files.')
        self.file_names = list()
        if data_files is not None:
            self.file_names = data_files
        elif data_name is not None:
            self.file_names.extend(glob.glob(os.path.join(self.data_path, self.data_name)))
        elif actions is not None:
            for action in actions:
                self.file_names.extend(glob.glob(os.path.join(self.data_path, f"*{action}*.npz")))

        # load data
        if self.b_pbar:
            print(f'Data loading... ({data_path})')
            print(f'Total {len(self.file_names)} files.')
        step = hp.step
        self.human_data = list()
        self.third_data = list()
        self.human_actions = list()
        for file in self.file_names:
            with np.load(file, allow_pickle=True) as data:
                self.human_data.append([norm_features(human, method='vector') for human in data['human_info']][::step])
                self.third_data.append(data['third_info'][::step])
                self.human_actions.append(data['human_action'][::step])

        # extract data
        self.inps = list()
        self.outs = list()
        pbar = tqdm(total=len(self.third_data)) if self.b_pbar else None
        for idx, third in enumerate(self.third_data):
            if all(v == 1.0 for v in third):
                continue
            for human_past_data, human_action in self.extract_data(
                    self.human_data[idx], self.human_actions[idx], self.third_data[idx], hp.user_pose_length):
                if human_action == 'None':
                    continue
                self.inps.append(human_past_data)
                # if human_action not in subaction_names:
                #     cur_action = 0
                # else:
                #     cur_action = subaction_names.index(human_action)
                cur_action = subaction_names.index(human_action)
                self.outs.append(cur_action)
            for f in range(hp.hold_last):
                self.inps.append(self.inps[-1])
                self.outs.append(self.outs[-1])
            pbar.update(1) if self.b_pbar else None
        pbar.close() if self.b_pbar else None
        print(f'Total {len(self.inps)} sequences.') if self.b_pbar else None

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
