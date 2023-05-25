import os
import glob
import copy
import random
import pickle
import numpy as np
from plyfile import PlyData

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as initer
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=1):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def adjust_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.0)
        except AttributeError:
            pass
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.0)
        except AttributeError:
            pass


def bn_momentum_adjust(m, momentum):
    if isinstance(m, nn.BatchNorm2d) or \
            isinstance(m, nn.BatchNorm1d):
        m.momentum = momentum


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    target[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def calc_victim_value(class_value, label, victim_class):
    values = []
    for lbl in victim_class:
        if label is None or (label == lbl).any():
            values.append(class_value[lbl])
    return np.mean(values)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (_ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, _BatchNorm):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def convert_to_syncbn(model):
    def recursive_set(cur_module, name, module):
        if len(name.split('.')) > 1:
            recursive_set(
                getattr(cur_module, name[:name.find('.')]), name[name.find('.')+1:], module)
        else:
            setattr(cur_module, name, module)
    from sync_bn import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, \
        SynchronizedBatchNorm3d
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            recursive_set(model, name, SynchronizedBatchNorm1d(
                m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm2d):
            recursive_set(model, name, SynchronizedBatchNorm2d(
                m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm3d):
            recursive_set(model, name, SynchronizedBatchNorm3d(
                m.num_features, m.eps, m.momentum, m.affine))


def lbl2rgb(label, names):
    """Convert label to rgb colors.
    label: [N]
    """
    from config import NAME2COLOR
    if len(names) == 13:
        colors = NAME2COLOR['S3DIS']
    else:
        colors = NAME2COLOR['ScanNet']
    rgb = np.zeros((label.shape[0], 3))
    uni_lbl = np.unique(label).astype(np.uint8)
    for lbl in uni_lbl:
        mask = (label == lbl)
        rgb[mask] = np.tile(np.array(
            colors[names[lbl]])[None, :], (mask.sum(), 1))
    return rgb


def convert2vis(xyz, label, names):
    """Assign color to each point according to label."""
    rgb = lbl2rgb(label, names) * 255.
    data = np.concatenate([xyz, rgb], axis=1)
    return data


def proc_pert(points, gt, pred, folder,
              names, part=False, ignore_label=255):
    """Process and save files for visulization in perturbation attack."""
    check_makedirs(folder)
    lbl2cls = {i: names[i] for i in range(len(names))}

    np.savetxt(os.path.join(folder, 'all_points.txt'), points, delimiter=';')
    gt_seg = convert2vis(points[gt != ignore_label, :3],
                         gt[gt != ignore_label], names)
    pred_seg = convert2vis(points[gt != ignore_label, :3],
                           pred[gt != ignore_label], names)
    np.savetxt(os.path.join(folder, 'gt.txt'),
               gt_seg, delimiter=';')
    np.savetxt(os.path.join(folder, 'pred.txt'),
               pred_seg, delimiter=';')
    if part:
        uni_lbl = np.unique(gt[gt != ignore_label]).astype(np.uint8)
        for lbl in uni_lbl:
            lbl = int(lbl)
            mask = (gt == lbl)
            sel_points = points[mask]
            mask = (gt[gt != ignore_label] == lbl)
            sel_seg = pred_seg[mask]
            np.savetxt(
                os.path.join(folder, '{}_{}_points.txt'.format(
                    lbl, lbl2cls[lbl])),
                sel_points, delimiter=';')
            np.savetxt(
                os.path.join(folder, '{}_{}_pred.txt'.format(
                    lbl, lbl2cls[lbl])),
                sel_seg, delimiter=';')


def proc_add(points, noise, gt, pred, noise_pred, folder,
             names, part=False, ignore_label=255):
    """Process and save files for visulization in adding attack."""
    check_makedirs(folder)
    lbl2cls = {i: names[i] for i in range(len(names))}

    np.savetxt(os.path.join(folder, 'all_points.txt'), points, delimiter=';')
    np.savetxt(os.path.join(folder, 'noise_points.txt'), noise, delimiter=';')
    gt_seg = convert2vis(points[gt != ignore_label, :3],
                         gt[gt != ignore_label], names)
    pred_seg = convert2vis(points[gt != ignore_label, :3],
                           pred[gt != ignore_label], names)
    noise_seg = convert2vis(noise[:, :3], noise_pred, names)
    np.savetxt(os.path.join(folder, 'gt.txt'),
               gt_seg, delimiter=';')
    np.savetxt(os.path.join(folder, 'pred.txt'),
               pred_seg, delimiter=';')
    np.savetxt(os.path.join(folder, 'noise_pred.txt'),
               noise_seg, delimiter=';')
    if part:
        uni_lbl = np.unique(gt[gt != ignore_label]).astype(np.uint8)
        for lbl in uni_lbl:
            lbl = int(lbl)
            mask = (gt == lbl)
            sel_points = points[mask]
            mask = (gt[gt != ignore_label] == lbl)
            sel_seg = pred_seg[mask]
            np.savetxt(
                os.path.join(folder, '{}_{}_points.txt'.format(
                    lbl, lbl2cls[lbl])),
                sel_points, delimiter=';')
            np.savetxt(
                os.path.join(folder, '{}_{}_pred.txt'.format(
                    lbl, lbl2cls[lbl])),
                sel_seg, delimiter=';')


def save_vis(pred_root, save_root, data_root):
    from config import CLASS_NAMES
    if 'S3DIS' in data_root:  # save Area5 data
        names = CLASS_NAMES['S3DIS']['other']
        gt_save = load_pickle(
            os.path.join(pred_root, 'gt_5.pickle'))['gt']
        pred_save = load_pickle(
            os.path.join(pred_root, 'pred_5.pickle'))['pred']
        assert len(gt_save) == len(pred_save)
        all_rooms = sorted(os.listdir(data_root))
        all_rooms = [
            room for room in all_rooms if 'Area_5' in room
        ]
        assert len(gt_save) == len(all_rooms)
        check_makedirs(save_root)
        for i, room in enumerate(all_rooms):
            points = np.load(os.path.join(data_root, room))[:, :6]
            folder = os.path.join(save_root, room[:-4])
            check_makedirs(folder)
            proc_pert(points, gt_save[i], pred_save[i],
                      folder, names, part=True)
    elif 'ScanNet' in data_root:  # save val set data
        names = CLASS_NAMES['ScanNet']['other']
        gt_save = load_pickle(
            os.path.join(pred_root, 'gt_val.pickle'))['gt']
        pred_save = load_pickle(
            os.path.join(pred_root, 'pred_val.pickle'))['pred']
        assert len(gt_save) == len(pred_save)
        data_file = os.path.join(
            data_root, 'scannet_val_rgb21c_pointid.pickle')
        file_pickle = open(data_file, 'rb')
        xyz_all = pickle.load(file_pickle)
        file_pickle.close()
        assert len(xyz_all) == len(gt_save)
        with open(os.path.join(
                data_root, 'meta_data/scannetv2_val.txt')) as fl:
            scene_id = fl.read().splitlines()
        assert len(scene_id) == len(gt_save)
        check_makedirs(save_root)
        for i in range(len(gt_save)):
            points = xyz_all[i][:, :6]
            folder = os.path.join(save_root, scene_id[i])
            check_makedirs(folder)
            proc_pert(points, gt_save[i], pred_save[i],
                      folder, names, part=True)


def save_vis_mink(pred_root, save_root, data_root):
    from config import CLASS_NAMES

    def load_data(file_name):
        plydata = PlyData.read(file_name)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']],
                          dtype=np.float32).T
        colors = np.array([data['red'], data['green'],
                           data['blue']], dtype=np.float32).T
        return np.concatenate([coords, colors], axis=1)

    if 'S3DIS' in data_root:  # save Area5 data
        names = CLASS_NAMES['S3DIS']['mink']
        gt_save = load_pickle(
            os.path.join(pred_root, 'gt_5.pickle'))['gt']
        pred_save = load_pickle(
            os.path.join(pred_root, 'pred_5.pickle'))['pred']
        assert len(gt_save) == len(pred_save)
        data_root = os.path.join(data_root, 'Area_5')
        all_rooms = sorted(os.listdir(data_root))
        assert len(all_rooms) == len(gt_save)
        check_makedirs(save_root)

        for i, room in enumerate(all_rooms):
            data = os.path.join(data_root, room)
            points = load_data(data)
            folder = os.path.join(
                save_root, 'Area_5_{}'.format(room[:-4]))
            check_makedirs(folder)
            proc_pert(points, gt_save[i], pred_save[i],
                      folder, names, part=True)
    elif 'ScanNet' in data_root:  # save val set
        names = CLASS_NAMES['ScanNet']['mink']
        gt_save = load_pickle(
            os.path.join(pred_root, 'gt_val.pickle'))['gt']
        pred_save = load_pickle(
            os.path.join(pred_root, 'pred_val.pickle'))['pred']
        assert len(gt_save) == len(pred_save)
        data_root = os.path.join(data_root, 'train')
        with open(os.path.join(
                data_root, 'scannetv2_val.txt'), 'r') as f:
            all_rooms = f.readlines()
        all_rooms = [room[:-1] for room in all_rooms]
        assert len(all_rooms) == len(gt_save)
        check_makedirs(save_root)

        for i, room in enumerate(all_rooms):
            data = os.path.join(data_root, room)
            points = load_data(data)
            folder = os.path.join(save_root, room[:-4])
            check_makedirs(folder)
            proc_pert(points, gt_save[i], pred_save[i],
                      folder, names, part=True)


def save_vis_from_pickle(pkl_root, save_root=None, room_idx=52,
                         room_name='scene0354_00'):
    names = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
        'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
        'curtain', 'refrigerator', 'showercurtain', 'toilet', 'sink',
        'bathtub', 'otherfurniture'
    ]
    data = load_pickle(pkl_root)
    points = data['data'][room_idx]
    pred = data['pred'][room_idx]
    gt = data['gt'][room_idx]
    if save_root is None:
        save_root = os.path.dirname(pkl_root)
    save_folder = os.path.join(save_root, room_name)
    proc_pert(points, gt, pred, save_folder, names, part=True)


def save_pickle(filename, dict_data):
    with open(filename, 'wb') as handle:
        pickle.dump(dict_data, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def load_s3dis_instance(folder, name2cls, load_name=['chair']):
    """Load S3DIS room in a Inst Seg format.
    Get each instance separately.

    If load_name is None or [], return all instances.
    Returns a list of [np.array of [N, 6], label]
    """
    cls2name = {name2cls[name]: name for name in name2cls.keys()}
    anno_path = os.path.join(folder, 'Annotations')
    points_list = []
    labels_list = []
    idx = 0
    files = glob.glob(os.path.join(anno_path, '*.txt'))
    files.sort()

    for f in files:
        cls = os.path.basename(f).split('_')[0]
        if cls not in name2cls.keys():
            cls = 'clutter'
        points = np.loadtxt(f)  # [N, 6]
        num = points.shape[0]
        points_list.append(points)
        labels_list.append((idx, idx + num, name2cls[cls]))
        idx += num

    # normalize points coords by minus min
    data = np.concatenate(points_list, 0)
    xyz_min = np.amin(data, axis=0)[0:3]
    data[:, 0:3] -= xyz_min

    # rearrange to separate instances
    if load_name is None or not load_name:
        load_name = list(name2cls.keys())
    instances = [
        [data[pair[0]:pair[1]], pair[2]] for pair in labels_list if
        cls2name[pair[2]] in load_name
    ]
    return instances


def cal_loss(pred, gold, smoothing=False, ignore_index=255):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(
            pred, gold, reduction='mean',
            ignore_index=ignore_index)

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()