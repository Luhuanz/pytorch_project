#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Ziyi Wu, Yingqi Wang
@Contact: dazitu616@gmail.com, yingqi-w19@mails.tsinghua.edu.cn
@File: main_semseg_scannet.py
@Time: 2022/7/30 7:49 PM
"""


import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch_scatter import scatter_sum
from tensorboardX import SummaryWriter
from plyfile import PlyData, PlyElement
from data import ScanNet
from model import DGCNN_semseg_scannet
from util import intersectionAndUnion, IOStream, str2bool, get_lr, adjust_lr, \
    weights_init, bn_momentum_adjust, cal_loss


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_semseg_scannet.py outputs'+'/'+args.exp_name+'/'+'main_semseg_scannet.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

def set_seed(seed=1):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser('Train point cloud semantic segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--data_root', type=str, default='data/ScanNet/',
                        help='path to data')
    parser.add_argument('--model_path', type=str, default='',
                        help='pretrained model weight')
    parser.add_argument('--split', type=str, default='val',
                        help='Split of data to evaluate on [default: val]',
                        choices=['val', 'test'])
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='Batch Size during training [default: 6]')
    parser.add_argument('--epoch', default=200, type=int,
                        help='Epoch to run [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Adam or SGD [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=8192,
                        help='Point Number [default: 8192]')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='Learning rate scheduler [default: cosine decay]')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='whether to evaluate pretrained weight [default: False]')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed [default: 1]')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization [default: ply]')
    parser.add_argument('--train_val', type=str2bool, default=False,
                        help='whether to train on both train set and val set [default: False]')

    return parser.parse_args()


def train(args, io):
    '''LOG'''
    if args.batch_size == -1:
        args.batch_size = 36
    io.cprint(str(args))

    root = args.data_root
    NUM_CLASSES = len(classes)
    NUM_POINT = args.npoint
    if NUM_POINT != 8192:
        io.cprint('Please check ScanNet num_points setting!')
    BATCH_SIZE = args.batch_size

    '''MODEL LOADING'''
    classifier = DGCNN_semseg_scannet(
        NUM_CLASSES, args.k, args.emb_dims, args.dropout)
    criterion = cal_loss
    # io.cprint(str(classifier))

    if args.sync_bn:
        io.cprint('Using Sync BN!')
        from util import convert_to_syncbn
        convert_to_syncbn(classifier)

    classifier = nn.DataParallel(classifier.cuda()).cuda()
    if args.sync_bn:
        from sync_bn import patch_replication_callback
        patch_replication_callback(classifier)

    if args.optimizer.lower() == 'Adam'.lower():
        optimizer = Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.decay_rate
        )
        io.cprint('Using Adam optimizer')
    else:
        optimizer = SGD(classifier.parameters(),
                        lr=args.learning_rate, momentum=0.9,
                        weight_decay=args.decay_rate)
        io.cprint('Using SGD optimizer')

    if args.lr_scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epoch, eta_min=1e-5)
        io.cprint('Using Cosine LR decay')
    else:
        lr_scheduler = StepLR(
            optimizer, step_size=args.step_size, gamma=args.lr_decay)
        io.cprint('Using Step LR decay')

    if args.model_path:
        io.cprint('Using pretrained model {}'.format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location='cpu')
        try:
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except KeyError:
            try:
                classifier.load_state_dict(checkpoint)
            except KeyError:
                classifier.module.load_state_dict(checkpoint)
    else:
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    '''DATA LOADING'''
    io.cprint('Loading data')
    if args.train_val:
        train_set = ScanNet(
                NUM_POINT, ['train', 'val'], root, NUM_CLASSES,
                transform=None, use_rgb=True)
    else:
        train_set = ScanNet(
                NUM_POINT, 'train', root, NUM_CLASSES,
                transform=None, use_rgb=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=min(32, BATCH_SIZE), pin_memory=True, drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    weights = None
    io.cprint("The number of training data is: %d" % len(train_set))

    '''CREATE DIR'''
    start_datetime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    logs_dir = "outputs/{}/".format(
        args.exp_name)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_writer = SummaryWriter(os.path.join(logs_dir, 'logs'))

    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0
    best_iou_epoch = -1

    for epoch in range(start_epoch, args.epoch + 1):
        '''Train on chopped scenes'''
        io.cprint('\n\n**** Epoch %d (%d/%s) ****' %
              (global_epoch, epoch, args.epoch))

        # BN momentum decay
        momentum = MOMENTUM_ORIGINAL * \
            (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        io.cprint('BN momentum updated to: %.4f' % momentum)
        classifier = classifier.apply(
            lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        num_batches = len(train_loader)
        total_correct = 0.
        total_seen = 0.
        loss_sum = 0.
        save_model = (epoch % 10 == 0)
        for i, data in tqdm(enumerate(train_loader),
                            total=len(train_loader), smoothing=0.9):
            points, target = data
            # TODO: data augmentation?
            # TODO: seems no rotation aug in original paper/code!
            # points = points.data.numpy()
            # points[:,:, :3] = provider.rotate_point_cloud_z(points[:,:, :3])
            # points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1)
            # TODO: loss weighting?
            # TODO: seems no loss weighting in original paper/code!
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                label = target.detach()
                keep_index = torch.where(label!=255)[0]
                preds = seg_pred.argmax(1)
                correct = (label[keep_index] == preds[keep_index]).sum(0).item()
            total_correct += correct
            total_seen += keep_index.shape[0]
            loss_sum += loss.item()
        mean_loss = loss_sum / num_batches
        mean_acc = total_correct / float(total_seen)
        lr = get_lr(optimizer)
        io.cprint('Training mean loss: %.4f' % (mean_loss))
        io.cprint('Training accuracy: %.4f' % (mean_acc))
        io.cprint('Learning rate: %.6f' % (lr))
        log_writer.add_scalar('train/loss', mean_loss, epoch)
        log_writer.add_scalar('train/accuracy', mean_acc, epoch)
        log_writer.add_scalar('train/lr', lr, epoch)

        # saving model when best mIoU or every 5 epochs
        if save_model:
            model_name = 'model_{}.pth'.format(epoch)
            savepath = os.path.join(logs_dir, 'models', model_name)
            io.cprint('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                # 'class_avg_iou': mIoU,
                'state_dict': classifier.state_dict(),
                'opt': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(state, savepath)

        global_epoch += 1
        lr_scheduler.step()
        torch.cuda.empty_cache()

def data_prepare(points, labels, num_point,
                 block_size=1.5, stride_rate=0.5):
    coord_min, coord_max = np.amin(points, axis=0)[
        :3], np.amax(points, axis=0)[:3]
    stride = block_size * stride_rate
    grid_x = int(np.ceil(float(
        coord_max[0] - coord_min[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil(float(
        coord_max[1] - coord_min[1] - block_size) / stride) + 1)
    data_room, label_room, index_room = \
        np.array([]), np.array([]), np.array([])
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_x = e_x - block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            point_idxs = np.where((points[:, 0] >= s_x - 1e-8) &
                                  (points[:, 0] <= e_x + 1e-8) &
                                  (points[:, 1] >= s_y - 1e-8) &
                                  (points[:, 1] <= e_y + 1e-8))[0]
            if point_idxs.size == 0:
                continue
            num_batch = int(np.ceil(point_idxs.size / num_point))
            point_size = int(num_batch * num_point)
            replace = False if (
                point_size - point_idxs.size <= point_idxs.size) else True
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            # np.random.shuffle(point_idxs)
            data_batch = points[point_idxs, :]  # [npoint, 6]
            normlized_xyz = np.zeros((point_size, 3))
            normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
            normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
            normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
            data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
            data_batch[:, 3:6] = data_batch[:, 3:6] / 255.  # RGB
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
            label_batch = labels[point_idxs]
            data_room = np.vstack([data_room, data_batch]) if \
                data_room.size else data_batch
            label_room = np.hstack([label_room, label_batch]) if \
                label_room.size else label_batch
            index_room = np.hstack([index_room, point_idxs]) if \
                index_room.size else point_idxs
    assert np.unique(index_room).size == labels.size
    return data_room, label_room, index_room

def evaluate(args, io):
    args.eval_batch_size = 6 * torch.cuda.device_count()
    io.cprint(str(args))

    model = DGCNN_semseg_scannet(
        args.classes, args.k, args.emb_dims, args.dropout)
    # io.cprint(str(model))
    model = nn.DataParallel(model).cuda()
    
    if not args.model_path:
        raise Exception("No pretrained model.")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        try:
            model.load_state_dict(checkpoint)
        except KeyError:
            model.module.load_state_dict(checkpoint)
    
    model.eval()
    if args.visu:
        visualization(model, args, io)
    else:
        if args.split=='val':
            eval_on_val(model, args, io)
        else:
            eval_on_test(model, args, io)

def eval_on_val(model, args, io):
    with torch.no_grad():
        model.eval()
        data_file = os.path.join(
            args.data_root, 'scannet_val_rgb21c_pointid_keep_unanno.pickle')
        file_pickle = open(data_file, 'rb')
        xyz_all = pickle.load(file_pickle)
        label_all = pickle.load(file_pickle)
        pickle.load(file_pickle)
        pickle.load(file_pickle)
        file_pickle.close()
        for i in range(len(label_all)):
            label = label_all[i] - 1
            label[label_all[i] == 0] = 255
            label_all[i] = label
        num_rooms = len(xyz_all)
        gt_all, pred_all = np.array([]), np.array([])
        for idx in range(num_rooms):
            points, gt = xyz_all[idx], label_all[idx].astype(np.int32)
            data_room, label_room, index_room = data_prepare(points, gt, args.npoint)
            data_room = torch.from_numpy(data_room).float()
            input = data_room.view(-1, args.npoint, data_room.shape[1]).transpose(2, 1)
            
            outputs = []
            for b in range(input.shape[0]//args.eval_batch_size + 1):
                input_b = input[b*args.eval_batch_size:(b+1)*args.eval_batch_size].cuda()
                if input_b.shape[0] == 0:
                    break
                if input_b.dim() < 3:
                    input_b = input_b.unsqueeze(0)
                output = model(input_b)[0]
                outputs.append(output.detach().cpu())
            
            #outputs = model(input.cuda()).detach().cpu()
            
            outputs = torch.cat(outputs)
            outputs = outputs.view(-1, args.classes)
        
            outputs_real = []
            for l in range(args.classes):
                outputs_real.append(scatter_sum(outputs[:,l], torch.LongTensor(index_room)).unsqueeze(1))
            outputs_real = torch.cat(outputs_real, dim=1)
            
            pred = torch.argmax(outputs_real, dim=1).numpy()
            
            intersection, union, target = intersectionAndUnion(pred, gt, args.classes, 255)
            mIoU = np.mean(intersection / (union + 1e-10))
            io.cprint('Room {}/{}, mIoU {:.4f}'.format(idx + 1, num_rooms, mIoU))
            pred_all = np.hstack([pred_all, pred]) if pred_all.size else pred
            gt_all = np.hstack([gt_all, gt]) if gt_all.size else gt
        
        intersection, union, target = intersectionAndUnion(pred_all, gt_all, args.classes, 255)
        iou_class = intersection / (union + 1e-10)
        mIoU = np.mean(iou_class)
        io.cprint('Val result: mIoU {:.4f}.'.format(mIoU))
        for i in range(args.classes):
            io.cprint('Class_{} Result: iou {:.4f}, name: {}.'.format(i, iou_class[i],  args.names[i]))
     
def eval_on_test(model, args, io):
    test_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
              10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    result_folder = os.path.join('outputs', args.exp_name, 'pred_result')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with torch.no_grad():
        model.eval()
        data_file = os.path.join(
            args.data_root, 'scannet_test_rgb21c_pointid_keep_unanno.pickle')
        file_pickle = open(data_file, 'rb')
        xyz_all = pickle.load(file_pickle)
        file_pickle.close()
        num_rooms = len(xyz_all)
        for idx in range(num_rooms):
            xyz = xyz_all[idx]
            label = np.zeros(len(xyz))
            data_room, label_room, index_room = data_prepare(xyz, label, args.npoint)
            with torch.no_grad():
                model.eval()
                data_room = torch.from_numpy(data_room).float()
                input = data_room.view(-1, args.npoint, data_room.shape[1]).transpose(2, 1)
                outputs = []
                for b in range(input.shape[0]//args.eval_batch_size + 1):
                    input_b = input[b*args.eval_batch_size:(b+1)*args.eval_batch_size].cuda()
                    if input_b.shape[0] == 0:
                        break
                    if input_b.dim() < 3:
                        input_b = input_b.unsqueeze(0)
                    output = model(input_b)[0]
                    outputs.append(output.detach().cpu())
                
                outputs = torch.cat(outputs)
                outputs = outputs.view(-1, args.classes)

                outputs_real = []
                for l in range(args.classes):
                    outputs_real.append(scatter_sum(outputs[:,l], torch.LongTensor(index_room)).unsqueeze(1))
                outputs_real = torch.cat(outputs_real, dim=1)
                
                pred = torch.argmax(outputs_real, dim=1).numpy()
                pred = [test_class[i+1] for i in pred]
                np.savetxt(os.path.join(result_folder, 'scene0{}_00.txt'.format(707+idx)), pred, fmt='%d')
    
def visualization(model, args, io):
    visualization_folder = os.path.join('outputs', args.exp_name, 'visualization')
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)
    visu = []
    if args.visu in ['all', 'train']:
        file_list = "prepare_data/scannetv2_train.txt"
        with open(file_list) as fl:
            scene_id = fl.read().splitlines()
        visu.extend(scene_id)
    if args.visu in ['all', 'val']:
        file_list = "prepare_data/scannetv2_val.txt"
        with open(file_list) as fl:
            scene_id = fl.read().splitlines()
        visu.extend(scene_id)
    if args.visu in ['all', 'test']:
        file_list = "prepare_data/scannetv2_test.txt"
        with open(file_list) as fl:
            scene_id = fl.read().splitlines()
        visu.extend(scene_id)
    if args.visu[:5] == 'scene':
        visu = args.visu.split(',')
    for i in visu:
        in_test_split = False
        try:
            in_test_split = not int(i[5:9])<=706
            if in_test_split:
                plydata = PlyData.read(os.path.join(args.data_root, 'scans_test', i, '{}_vh_clean_2.ply'.format(i)))
            else:
                plydata = PlyData.read(os.path.join(args.data_root, 'scans', i, '{}_vh_clean_2.labels.ply'.format(i)))
        except:
            io.cprint('Can not find file {}_vh_clean_2.ply.'.format(i))
            continue
        xyz = np.asarray([[plydata.elements[0].data[i][0], plydata.elements[0].data[i][1], plydata.elements[0].data[i][2], plydata.elements[0].data[i][3], plydata.elements[0].data[i][4], plydata.elements[0].data[i][5]] for i in range(len(plydata.elements[0].data))])
        if in_test_split:
            label = np.zeros(len(xyz))
        else:
            label_cov = [255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 12, 255, 13, 255, 255, 255, 255, 255, 255, 255, 14, 255, 255, 255, 15, 255, 255, 255, 255, 16, 17, 255, 18, 255, 255, 19, 255]
            label = np.asarray([label_cov[plydata.elements[0].data[i][7]] for i in range(len(plydata.elements[0].data))])
        data_room, label_room, index_room = data_prepare(xyz, label, args.npoint)
        with torch.no_grad():
            model.eval()
            data_room = torch.from_numpy(data_room).float()
            input = data_room.view(-1, args.npoint, data_room.shape[1]).transpose(2, 1)
            outputs = []
            for b in range(input.shape[0]//args.eval_batch_size + 1):
                input_b = input[b*args.eval_batch_size:(b+1)*args.eval_batch_size].cuda()
                if input_b.shape[0] == 0:
                    break
                if input_b.dim() < 3:
                    input_b = input_b.unsqueeze(0)
                output = model(input_b)[0]
                outputs.append(output.detach().cpu())
            
            outputs = torch.cat(outputs)
            outputs = outputs.view(-1, args.classes)

            outputs_real = []
            for l in range(args.classes):
                outputs_real.append(scatter_sum(outputs[:,l], torch.LongTensor(index_room)).unsqueeze(1))
            outputs_real = torch.cat(outputs_real, dim=1)
            
            pred = torch.argmax(outputs_real, dim=1).numpy()
            
            points_to_save = [(xyz[i, 0], xyz[i, 1], xyz[i, 2], label2color[pred[i]][0], label2color[pred[i]][1], label2color[pred[i]][2]) for i in range(xyz.shape[0])]
            if args.visu_format.lower()=='ply':
                vertex = np.array(points_to_save, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
                el = PlyElement.describe(vertex, 'vertex')
                PlyData([el]).write(os.path.join(visualization_folder, '{}.ply'.format(i)))
            elif args.visu_format.lower()=='txt':
                np.savetxt(os.path.join(visualization_folder, '{}.txt'.format(i)), points_to_save)
            else:
                io.cprint('Visulization format is not supported. Please use ply or txt.')
                continue
            io.cprint('Visulize {} successfully!'.format(i))
            if not in_test_split:
                points_to_save = [(xyz[i, 0], xyz[i, 1], xyz[i, 2], label2color[label[i]][0], label2color[label[i]][1], label2color[label[i]][2]) for i in range(xyz.shape[0])]
                if args.visu_format.lower()=='ply':
                    vertex = np.array(points_to_save, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
                    el = PlyElement.describe(vertex, 'vertex')
                    PlyData([el]).write(os.path.join(visualization_folder, '{}_gt.ply'.format(i)))
                elif args.visu_format.lower()=='txt':
                    np.savetxt(os.path.join(visualization_folder, '{}.txt'.format(i)), points_to_save)
                else:
                    io.cprint('Visulization format is not supported. Please use ply or txt.')
                    continue

if __name__ == '__main__':
    args = parse_args()
    _init_()
    io = IOStream('outputs/' + args.exp_name + '/run.log')
    
    set_seed(args.seed)
    cudnn.benchmark = True
    # TODO: Sync BN?
    args.sync_bn = (torch.cuda.device_count() > 1)

    io.cprint('Experiment on ScanNet dataset!')
    classes = [
        'wall', 'floor', 'cabinet', 'bed',
        'chair', 'sofa', 'table', 'door',
        'window', 'bookshelf', 'picture', 'counter',
        'desk', 'curtain', 'refrigerator', 'showercurtain',
        'toilet', 'sink', 'bathtub', 'otherfurniture'
    ]
        
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    label2color = {}
    class2color = {
        'wall': [174, 199, 232],
        'floor': [152, 223, 138],
        'cabinet': [31, 119, 180],
        'bed': [255, 187, 120],
        'chair': [188, 189, 34],
        'sofa': [140, 86, 75],
        'table': [255, 152, 150],
        'door': [214, 39, 40],
        'window': [197, 176, 213],
        'bookshelf': [148, 103, 189],
        'picture': [196, 156, 148],
        'counter': [23, 190, 207],
        'desk': [247, 182, 210],
        'curtain': [219, 219, 141],
        'refrigerator': [255, 127, 14],
        'showercurtain': [158, 218, 229],
        'toilet': [44, 160, 44],
        'sink': [112, 128, 144],
        'bathtub': [227, 119, 194],
        'otherfurniture': [82, 84, 163],
    }
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat
        label2color[i] = class2color[cat]
    label2color[255] = [0, 0, 0]
    args.classes = len(classes)
    args.names = classes
    args.label2color = label2color
    if not args.eval:
        train(args, io)
    else:
        evaluate(args, io)
