import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

input_dim = 3

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list) = zip(*data)
    # print(obs_seq_list[3])
    # print(obs_seq_list[1].size())


    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.stack(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.stack(pred_seq_list, dim=0).permute(2, 0, 1)
    out = [
        obs_traj, pred_traj
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory dataset"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        seq_list = []
        for path in all_files:
            data = read_file(path, delim)
            print(data[:, 1])
            unique_index = np.unique(data[:, 1]).tolist()
            for index in unique_index:
                index_data = data[index == data[:, 1], :]

                # 判断是否连续
                if index_data[0, 0] + 60 * (np.size(index_data, 0) - 1) != index_data[-1, 0]:
                    continue

                # 判断长度是否够
                if np.size(index_data, 0) < self.seq_len:
                    continue

                for i in range(np.size(index_data, 0) - self.seq_len + 1):
                    choose_data = index_data[i:(i + self.seq_len), :]
                    choose_data = np.transpose(choose_data[:, 2:])

                    seq_list.append(np.expand_dims(choose_data, 0))

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        out = [
            self.obs_traj[index, :, :], self.pred_traj[index, :, :]
        ]
        return out
