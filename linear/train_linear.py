import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from loader_linear import data_loader
from losses_linear import l2_loss
from losses_linear import displacement_error, final_displacement_error

from models_linear import LinearModel
from utils.utils import int_tuple, bool_flag, get_total_norm
from utils.utils import relative_to_abs, get_dset_path


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./run/usa_random_10_10_linear_1')

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='asia4_smooth', type=str)
parser.add_argument('--delim', default='tab')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=2, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default='0', type=str)

def init_weights(net):
    if type(net) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(net.weight)
        net.bias.data.fill_(0.01)  # tots els bias a 0.01

# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    global global_step
    global_step = 13086


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    print(args)
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    linear = LinearModel(
        seq_len=args.pred_len,
        use_cuda=args.use_gpu)

    linear.apply(init_weights)
    linear.type(float_dtype).train()
    logger.info('Here is the linear:')
    logger.info(linear)

    optimizer = optim.Adam(linear.parameters(), lr=args.g_learning_rate)

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        linear.load_state_dict(checkpoint['state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint dataset structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'state': None,
            'optim_state': None,
            'best_state': None,
            'best_t': None,
            'layer1.weight':None,
            'layer2.weight': None,
            "layer1.bias":None,
            "layer2.bias": None
        }
    t0 = None
    while t < args.num_iterations:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            generator_step(args, batch, linear, optimizer)
            # checkpoint['norm_g'].append(
            #     get_total_norm(lstm.parameters())
            # )

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
            #     for k, v in sorted(losses.items()):
            #         logger.info('  [D] {}: {:.7f}'.format(k, v))
            #         checkpoint['losses'][k].append(v)
            #     checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, linear, is_train=False
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, linear, limit=True, is_train=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.7f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.7f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = linear.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['state'] = linear.state_dict()
                checkpoint['optim_state'] = optimizer.state_dict()

                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')


            t += 1
            if t >= args.num_iterations:
                break


def generator_step(
    args, batch, linear, optimizer
):
    if args.use_gpu == 1:
        batch = [tensor.cuda() for tensor in batch]
    else:
        batch = [tensor for tensor in batch]
    # batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt) = batch
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = linear(obs_traj)

    pred_traj_fake = generator_out

    losses = l2_loss(pred_traj_fake, pred_traj_gt, mode='raw')
    loss += torch.sum(losses)

    # writer.add_scalar('loss', loss.item(), global_step=global_step)

    optimizer.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            linear.parameters(), args.clipping_threshold_g
        )
    optimizer.step()

    return loss.item


def check_accuracy(
    args, loader, generator, limit=False, is_train=True
):
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            # batch = [tensor.cuda() for tensor in batch]
            if args.use_gpu == 1:
                batch = [tensor.cuda() for tensor in batch]
            else:
                batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt) = batch

            pred_traj_fake = generator(obs_traj)

            g_l2_loss_abs= cal_l2_losses(pred_traj_gt, pred_traj_fake)

            ade = cal_ade(pred_traj_gt, pred_traj_fake)

            fde = cal_fde(pred_traj_gt, pred_traj_fake)

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            total_traj += pred_traj_gt.size(1)
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / total_traj

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj



    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_fake
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, mode='sum'
    )
    return g_l2_loss_abs


def cal_ade(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)

    return ade


def cal_fde(pred_traj_gt, pred_traj_fake):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])

    return fde


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
