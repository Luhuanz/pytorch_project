import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')# 设定所采用数据集的名字为‘coco’
    parser.add_argument('--coco_path', help='Path to COCO directory')# 设定coco数据集所在的路径
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')# 设定自己的数据集中train部分图片的路径
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')# 设定自己的数据集中图片类别信息的路径
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')# 设定自己的数据集中val部分图片的路径

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50) # 设定所采用resnet的深度
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)# 设定模型所跑的epoch的总数

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        # 读入训练的数据，并通过Normalizer(), Augmenter(), Resizer()三个工具对数据进行预处理
        dataset_train = CocoDataset(parser.coco_path, set_name='val2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        # 读入验证的数据，相较于train的数据少了Augmenter()的预处理部分
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
# AspectRatioBasedSampler函数的作用为将dataset_train的数据变成以batch_size一个一个group
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)  # 2599次
    # collater返回图片[batch_size,h,w,c]，标记[batchsize,?,5]（其中？由有最多标记数目的图片决定，无用的标记以[-1,-1,-1,-1,-1]表示）

# dataset：Dataset类型，从其中加载数据
# batch_size：int，可选。每个batch加载多少样本
# shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
# sampler：Sampler，可选。从数据集中采样样本的方法。
# num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
# collate_fn：callable，可选。
# pin_memory：bool，可选
# drop_last：bool，可选。True表示如果最后剩下不完全的batch,丢弃。False表示不丢弃
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model  读入模型，且需要有预训练的数据
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    # if torch.cuda.is_available(): # 用多个GPU
    #     retinanet = torch.nn.DataParallel(retinanet).cuda()
    # else:
    #     retinanet = torch.nn.DataParallel(retinanet)
    # # 将retinanet设定为训练状态
    retinanet.training = True
    # 优化器为Adam,learning_rate为0.5（实际论文中为SGD）
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    # class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
    #                                                  verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # optimer指的是网络的优化器
    # mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
    # factor 学习率每次降低多少，new_lr = old_lr * factor
    # patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
    # verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
    # threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
    # cooldown： 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
    # min_lr,学习率的下限
    # eps ，适用于lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad() # 将上一步中的Gradient置0

                if torch.cuda.is_available():# 将图片和annotation输入模型，获得两个loss
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                # 将两个loss做平均
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                # 求和
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue
                # 求导
                loss.backward()
                # 实现了Clipping Gradient，避免梯度爆炸的出现
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                # 应用gradient到模型的变量上去
                optimizer.step()
                # 记录loss
                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            # 用val集来评价模型性能
            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        # 更新优化器的参数
        scheduler.step(np.mean(epoch_loss))
        # 存下模型
        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
