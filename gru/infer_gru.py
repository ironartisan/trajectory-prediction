import argparse
import os
import torch
import pdb
import traceback
import torch.nn as nn
# import torchsnoop

from attrdict import AttrDict

from loader_gru import data_loader
from models_gru import GRUModel
from utils.utils import relative_to_abs, get_dset_path

from train_gru import cal_ade, cal_fde


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--use_cuda', default=1, type=int)


class Infer(object):
    def __init__(self, use_cuda=0):
        self.use_cuda = use_cuda

    
    def get_accuracy(self):
        return self.metrics_train, self.metrics_val

    def load_model(self, path):
        # torch.load最后返回的是一个dict，里面包含了保存模型时的一些参数和模型
        checkpoint = torch.load(path, map_location='cpu')
        self.gru = self.get_checkpoint(checkpoint)
        # AttrDict是根据参数中的dict内容生成一个更加方便访问的dict实例
        self.args = AttrDict(checkpoint['args'])
        train_path = get_dset_path(self.args.dataset_name, "train")
        test_path = get_dset_path(self.args.dataset_name, "test")
        self.args.batch_size = 1
        _, self.loader = data_loader(self.args, train_path)
        _, self.test_loader = data_loader(self.args, test_path)


        self.metrics_val = checkpoint['metrics_val']
        self.metrics_train = checkpoint['metrics_train']
    def infer(self):
        with torch.no_grad():
            # print(len(self.loader))
            for batch in self.loader:
                if self.use_cuda == 1:
                    batch = [tensor.cuda() for tensor in batch]
                else:
                    batch = [tensor for tensor in batch]
                (obs_traj, pred_traj_gt) = batch

                # [8, 4, 2]
                pred_traj_fake = self.gru(obs_traj)

                obs_traj = obs_traj.cpu().numpy()
                pred_traj_fake = pred_traj_fake.cpu().numpy()
                pred_traj_gt = pred_traj_gt.cpu().numpy()

                return obs_traj, pred_traj_fake, pred_traj_gt

    def predict(self, obs_traj, pred_traj_gt):
        pred_traj_fake = self.gru(obs_traj)

        ade = cal_ade(pred_traj_gt, pred_traj_fake)
        fde = cal_fde(pred_traj_gt, pred_traj_fake)

        pred_traj_fake = pred_traj_fake.cpu().detach().numpy()

        ade = ade.cpu().detach().numpy()
        fde = fde.cpu().detach().numpy()

        return pred_traj_fake, ade, fde

    # @torchsnooper.snoop()
    def check_accuracy(self, loader_type='test', limit=True):
        if loader_type == 'test':
            loader = self.test_loader
        else:
            loader = self.loader

        args = self.args
        gru = self.gru
        
        metrics = {}
        disp_error, f_disp_error = [], []
        total_traj = 0
        with torch.no_grad():
            for batch in loader:
                # batch = [tensor.cuda() for tensor in batch]
                if args.use_gpu == 1:
                    batch = [tensor.cuda() for tensor in batch]
                else:
                    batch = [tensor for tensor in batch]
                (obs_traj, pred_traj_gt) = batch
                # obs_traj = obs_traj.cpu()

                pred_traj_fake = gru(obs_traj)

                ade = cal_ade(pred_traj_gt, pred_traj_fake)

                fde = cal_fde(pred_traj_gt, pred_traj_fake)

                disp_error.append(ade.item())
                f_disp_error.append(fde.item())

                total_traj += pred_traj_gt.size(1)
                if limit and total_traj >= args.num_samples_check:
                    break

        metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
        metrics['fde'] = sum(f_disp_error) / total_traj
        return metrics

    def get_one_data(self):
        with torch.no_grad():
            # print(len(self.loader))
            for batch in self.loader:
                if self.use_cuda == 1:
                    batch = [tensor.cuda() for tensor in batch]
                else:
                    batch = [tensor for tensor in batch]
                (obs_traj, pred_traj_gt) = batch

                return obs_traj, pred_traj_gt

    def get_checkpoint(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        gru = GRUModel(
            seq_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_cuda=args.use_gpu)
        gru.load_state_dict(checkpoint['best_state'])
        if self.use_cuda == 1:
            gru.cuda()
        gru.eval()
        return gru


if __name__ == '__main__':
    args = parser.parse_args()
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    path = paths[0]

    print(path)
    infer = Infer(args)
    infer.load_model(path)

    # obs_traj, pred_traj_fake, pred_traj_gt = infer.infer()
    metrics = infer.check_accuracy('test', limit=True)
    print('linear test ade is %f' % metrics['ade'])
    print('linear test fde is %f' % metrics['fde'])




