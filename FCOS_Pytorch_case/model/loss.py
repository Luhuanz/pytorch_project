"""
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-10-06
@Email: xxxmy@foxmail.com
"""

import torch
import torch.nn as nn
from model.config import DefaultConfig


def coords_fmap2orig(feature, stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    h, w = feature.shape[1:3]  # 得到该层输出结果的h和w
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)  # 得到该层点对应左上角的x和y的坐标
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])  # 通过meshgrid的方式把所有框左上角的坐标找出来
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2  # 最后都位移到框的中心去
    return coords


class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides  # 在config file中定义了每一个特征层对应的stride为多少
        self.limit_range = limit_range  # 在config file中定义了每一个特征层对应的框的范围为多少
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        """
        inputs
        [0]list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        [1]gt_boxes [batch_size,m,4]  FloatTensor
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        """
        cls_logits, cnt_logits, reg_preds = inputs[0]  # input[0]为FCOS网络的预测结果，现在分解为这三项
        gt_boxes = inputs[1]  # 第二项代表实际的gt boxes
        classes = inputs[2]  # 第三项代表实际的gt boxes的类别
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):  # 界限来分层来进行目标的一个转换
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]  # 将一层的目标打包成一个list
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])  # 根据输入的gt boxes找出每一个坐标点对应的回归目标为多少
            cls_targets_all_level.append(level_targets[0])  # 将这几个目标存起来
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), torch.cat(
            reg_targets_all_level, dim=1)  # 将不同层的目标合并起来

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits, cnt_logits, reg_preds = out  # 将该层中FCOS的预测结果分解成三项
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]  # 得到此时gt boxes的数量

        cls_logits = cls_logits.permute(0, 2, 3, 1)  # [batch_size,h,w,class_num]
        coords = coords_fmap2orig(cls_logits, stride).to(device=gt_boxes.device)  # [h*w,2]  找到特征图上的点对应实际图片中的位置

        cls_logits = cls_logits.reshape((batch_size, -1, class_num))  # [batch_size,h*w,class_num]
        cnt_logits = cnt_logits.permute(0, 2, 3, 1)  # 全部把代表预测结果的维度移动到最后一维，并平h和w的维度
        cnt_logits = cnt_logits.reshape((batch_size, -1, 1))
        reg_preds = reg_preds.permute(0, 2, 3, 1)
        reg_preds = reg_preds.reshape((batch_size, -1, 4))

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]  # 计算中心点和gt box四条边的距离
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)  # [batch_size,h*w,m,4]  把这四个距离记录下来，顺序为left,top,right,bottom

        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])  # [batch_size,h*w,m]

        off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]  找到每一个坐标对应的框的最大值和最小值
        off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]

        mask_in_gtboxes = off_min > 0  # 统计那些坐标在gt boxes内的
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])  # 通过limit_range来计算那些属于该level的坐标点

        radiu = stride * sample_radiu_ratio  # 控制对应的坐标点应在gt boxes中心的附近
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2  # 找到gt boxes的中心
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]  # 计算gt boxes的中心与中心点在原图位置之间的距离
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)  # [batch_size,h*w,m,4]
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]  # 计算上下左右最选的距离为多少
        mask_center = c_off_max < radiu  # 只选择小于1.5倍stride的点

        mask_pos = mask_in_gtboxes & mask_in_level & mask_center  # [batch_size,h*w,m]  # 最后将上述三个标准合并下，就得到了每一个gt boxes可能对应的点是哪些

        areas[~mask_pos] = 99999999  # 非目标点的面积被设定为无穷大
        areas_min_ind = torch.min(areas, dim=-1)[1]  # [batch_size,h*w]  找到每一个gt boxes对应的中心点预测结果面积与那个gt boxes更接近
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.uint8).scatter_(-1, areas_min_ind.unsqueeze(dim=-1),
                                                                                   1)]  # [batch_size*h*w,4]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))  # [batch_size,h*w,4]  # 输出每一个坐标点最后应该实现的目标为多少

        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]  # [batch_size,h*w,m]
        cls_targets = classes[
            torch.zeros_like(areas, dtype=torch.uint8).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))  # [batch_size,h*w,1]  同理找到对应的应有的target为多少

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])  # [batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(
            dim=-1)  # [batch_size,h*w,1]  # 根据公式计算此时的centerness目标

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        assert cnt_targets.shape == (batch_size, h_mul_w, 1)

        # process neg coords
        mask_pos_2 = mask_pos.long().sum(dim=-1)  # [batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2 = mask_pos_2 >= 1  # 找到那些不属于任何一个gt boxes的坐标
        assert mask_pos_2.shape == (batch_size, h_mul_w)
        cls_targets[~mask_pos_2] = 0  # [batch_size,h*w,1]  将他们的类别分为0
        cnt_targets[~mask_pos_2] = -1  # cnt和reg都为-1,不用计算
        reg_targets[~mask_pos_2] = -1

        return cls_targets, cnt_targets, reg_targets


def compute_cls_loss(preds, targets, mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, class_num])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)  # [batch_size,sum(_h*_w),class_num] 将pred按照target的构成方式调整为一样的大小
    assert preds.shape[:2] == targets.shape[:2]
    loss = []
    for batch_index in range(batch_size):  # 分不同的图片分别计算loss
        pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
        target_pos = targets[batch_index]  # [sum(_h*_w),1]
        target_pos = (torch.arange(1, class_num + 1, device=target_pos.device)[None,
                      :] == target_pos).float()  # sparse-->onehot  将原有的目标转换为onehot的形式
        loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))  # 在cls的loss计算中，用了focal loss
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_cnt_loss(preds, targets, mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    mask = mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),1]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,]
        assert len(pred_pos.shape) == 1
        loss.append(
            nn.functional.binary_cross_entropy_with_logits(input=pred_pos, target=target_pos, reduction='sum').view(1))  # 在cnt的loss计算在仅用了交叉熵
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_reg_loss(preds, targets, mask, mode='giou'):
    '''
    Args  
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),4]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,4]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,4]
        assert len(pred_pos.shape) == 2
        if mode == 'iou':
            loss.append(iou_loss(pred_pos, target_pos).view(1))  # 在reg的loss计算中运用了两种不同的iou
        elif mode == 'giou':
            loss.append(giou_loss(pred_pos, target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def iou_loss(preds, targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt = torch.min(preds[:, :2], targets[:, :2])
    rb = torch.min(preds[:, 2:], targets[:, 2:])
    wh = (rb + lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    iou = overlap / (area1 + area2 - overlap)
    loss = -iou.clamp(min=1e-6).log()
    return loss.sum()


def giou_loss(preds, targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (rb_min + lt_min).clamp(min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap / union

    lt_max = torch.max(preds[:, :2], targets[:, :2])
    rb_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = (rb_max + lt_max).clamp(0)
    G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]

    giou = iou - (G_area - union) / G_area.clamp(1e-10)
    loss = 1. - giou
    return loss.sum()


def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    '''
    preds = preds.sigmoid()
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
    return loss.sum()


class LOSS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs):
        """
        inputs list
        [0]preds:  ....
        [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        """
        preds, targets = inputs
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)  # [batch_size,sum(_h*_w)]
        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask_pos).mean()  # []  # 分别计算三种loss
        cnt_loss = compute_cnt_loss(cnt_logits, cnt_targets, mask_pos).mean()
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask_pos).mean()
        if self.config.add_centerness:  # 返回单个loss和相加后的loss
            total_loss = cls_loss + cnt_loss + reg_loss
            return cls_loss, cnt_loss, reg_loss, total_loss
        else:
            total_loss = cls_loss + reg_loss + cnt_loss * 0.0
            return cls_loss, cnt_loss, reg_loss, total_loss


if __name__ == "__main__":
    loss = compute_cnt_loss([torch.ones([2, 1, 4, 4])] * 5, torch.ones([2, 80, 1]),
                            torch.ones([2, 80], dtype=torch.uint8))
    print(loss)
