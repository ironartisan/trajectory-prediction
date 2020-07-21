import torch
import random
from math import radians, cos, sin, asin, sqrt, pi


lat_min, lat_field, long_min, long_field, alt_min, alt_field = 3.1933607061071148,50.80615587860828,78.42022293891259,56.574887353137626,0.0,43734.576275625324


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input dataset.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.ones_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / batch
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)

    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos

    # loss[:, 0] = loss[:, 0] * lat_field + lat_min
    # loss[:, 1] = loss[:, 1] * long_field + long_min
    # loss[:, 2] = loss[:, 2] * alt_field + alt_min

    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def geodistance(lat1, lng1, alt1, lat2, lng2, alt2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    ground_distance = 2 * asin(sqrt(a)) * 6371 * 1000 # 地球平均半径，6371km
    ground_distance = round(ground_distance / 1000, 3)

    distance = sqrt((alt2 - alt1) ** 2 + ground_distance ** 2)
    return distance

def geo_displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    real_pred_traj = pred_traj.permute(1, 0, 2)
    real_pred_traj_gt = pred_traj_gt.permute(1, 0, 2)

    real_pred_traj[:, :, 0] = real_pred_traj[:, :, 0] * lat_field + lat_min
    real_pred_traj_gt[:, :, 0] = real_pred_traj_gt[:, :, 0] * lat_field + lat_min

    real_pred_traj[:, :, 1] = real_pred_traj[:, :, 1] * long_field + long_min
    real_pred_traj_gt[:, :, 1] = real_pred_traj_gt[:, :, 1] * long_field + long_min

    real_pred_traj[:, :, 2] = real_pred_traj[:, :, 2] * alt_field + alt_min
    real_pred_traj_gt[:, :, 2] = real_pred_traj_gt[:, :, 2] * alt_field + alt_min

    angle_distance = make_radians(real_pred_traj_gt[:, :, :2]) - make_radians(real_pred_traj[:, :, :2])
    temp = torch.sin(angle_distance[:, :, 0] / 2) ** 2 + torch.cos(real_pred_traj[:, :, 0]) * torch.cos(real_pred_traj_gt[:, :, 0]) * torch.sin(angle_distance[:, :, 1] / 2) ** 2
    ground_distance = 2 * torch.asin(torch.sqrt(temp)) * 6371 * 1000

    loss = torch.sqrt((real_pred_traj_gt[:, :, 2] - real_pred_traj[:, :, 2]) ** 2 + ground_distance ** 2)

    if consider_ped is not None:
        loss = loss.sum(dim=1) * consider_ped
    else:
        loss = loss.sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def geo_final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    real_pred_pos = torch.clone(pred_pos)
    real_pred_pos_gt = torch.clone(pred_pos_gt)

    real_pred_pos[:, 0] = pred_pos[:, 0] * lat_field + lat_min
    real_pred_pos_gt[:, 0] = pred_pos_gt[:, 0] * lat_field + lat_min

    real_pred_pos[:, 1] = pred_pos[:, 1] * long_field + long_min
    real_pred_pos_gt[:, 1] = pred_pos_gt[:, 1] * long_field + long_min

    real_pred_pos[:, 2] = pred_pos[:, 2] * alt_field + alt_min
    real_pred_pos_gt[:, 2] = pred_pos_gt[:, 2] * alt_field + alt_min

    angle_distance = make_radians(real_pred_pos_gt[:, :2]) - make_radians(real_pred_pos[:, :2])
    temp = torch.sin(angle_distance[:, 0] / 2) ** 2 + torch.cos(real_pred_pos[:, 0]) * torch.cos(real_pred_pos_gt[:, 0]) * torch.sin(angle_distance[:, 1] / 2) ** 2
    ground_distance = 2 * torch.asin(torch.sqrt(temp)) * 6371 * 1000

    loss = torch.sqrt((real_pred_pos_gt[:, 2] - real_pred_pos[:, 2]) ** 2 + ground_distance ** 2)

    if consider_ped is not None:
        loss = loss * consider_ped

    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def make_radians(temp):
    return temp * pi / 180