import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b): # 用于计算IoU的函数
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        # 重新将anchors的值从左上坐标，右下坐标）转为（中心坐标，宽高）格式
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):# 对于batch_size中的每一张图片，做以下处理

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]# 取bbox_annotation的值不为-1的框

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)# 将类别数值规范到[1e-4, 1.0 - 1e-4]，避免取对数时候出现问题

            if bbox_annotation.shape[0] == 0:# 将类别数值规范到[1e-4, 1.0 - 1e-4]，避免取对数时候出现问题
                if torch.cuda.is_available():# 分有没有GPU的两种情况
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue
          #  接着计算所有anchor与真实框的IOU大小
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            # 找到所有anchor IOU最大的真实框的索引以及该IOU大小
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for classification   # 开始计算两个子网络的损失
            targets = torch.ones(classification.shape) * -1  #初始全为-1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0 # IOU<0.4为负样本，记为0


            positive_indices = torch.ge(IoU_max, 0.5)  # IOU>=0.5为正样本，找到index

            num_positive_anchors = positive_indices.sum() # 正样本个数

            assigned_annotations = bbox_annotation[IoU_argmax, :]# 通过IoU_argmax找到对应的实际annotations为哪一个（anchor_nums,4）

            # 计算分类子网络的损失
            targets[positive_indices, :] = 0# 将targets中正样本对应的类别全赋值为0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1# 通过查assigned_annotations第5位上的标签信息，实现one-hot的效果

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha
            # torch.where的作用是[1]满足则[2]，不满足则[3]
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)# 正样本用alpha，负样本用1-alpha
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification) # 正样本用1-classification ，负样本用classification
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)# 对应文中的alpha×(1-classification)^gamma

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))# 普通的Balanced Cross Entropy公式

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce # 将focal_weight与普通的Balanced Cross Entropy就可以得到Focal Loss

            if torch.cuda.is_available():# 如果targets不存在（为-1），此时的cls_loss置为常数0
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))# 将classification loss求并除以num_positive_anchors的数目

            # compute the loss for regression 计算回归框子函数的损失

            if positive_indices.sum() > 0:  # 当存在positive_indices的时候进行计算
                assigned_annotations = assigned_annotations[positive_indices, :]# 找到当存在positive_indices的时候进行计算对应的assigned_annotations
                # 找到positive_indices对应的anchors的四个值
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # 重新将assigned_annotations的值从左上坐标，右下坐标）转为（中心坐标，宽高）格式
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1  最小框的长宽不会小于1个像素点
                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                # 结合assigned_annotations（实际的）和anchor计算regression应该预测的值为多少（这部分和SSD的过程一致）
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():# 将targets的值做一个扩大，应该是为了扩大regression输出值拟合的范围
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :]) # 取实际与预测的相对误差

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


