import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """"从(X，mask)元组列表中构建小批量张量。掩码输入
    Args:
        data: (batch_size)长度的元组列表(X，y)
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: 全局固定序列长度。用于需要固定长度输入的体系结构，在其中批次长度不能动态变化。较长的序列被剪辑，较短的序列被填充为0
    Returns:
        X: 形状为(batch_size，padded_length，feat_dim)的torch张量，掩码特征(输入)
            targets：形状为(batch_size，padded_length，feat_dim)的torch张量，未掩码特征(输出)
        target_masks：形状为(batch_size，padded_length，feat_dim)的布尔torch张量， 0表示要预测的掩码值，1表示不受影响/"活动"特征值
        padding_masks:
    """

    batch_size = len(data) # # 获取batch_size
    features, labels = zip(*data)# 将data中的features和labels分别存储到两个列表中

    # 堆叠和填充特征和掩码(将2D转换为3D张量，即添加批次维度)
    lengths = [X.shape[0] for X in features]  # 每个时间序列的原始序列长度
    if max_len is None:   # 如果max_len为None，则将其设为lengths中的最大值
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len) # 将长度限制在max_len范围内
        X[i, :end, :] = features[i][:end, :] # 将特征填充到X张量中

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep 填充掩码

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Normalizer(object):
    """
  数据标准化类，用于对DataFrame进行标准化处理，可以按照所有样本的数据或者按照单个样本进行标准化。
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
          Args:
              norm_type: 标准化类型，可选"standardization"（标准化）, "minmax"（最大最小标准化）,
                         "per_sample_std"（每个样本独立的标准化）, "per_sample_minmax"（每个样本独立的最大最小标准化）。
              mean, std, min_val, max_val: 可选参数，用于输入预先计算好的均值、标准差、最小值和最大值。
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
           对输入的DataFrame进行标准化处理。
               df: 需要标准化的DataFrame。
           Returns: 标准化后的DataFrame。
           """

        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

#使用线性插值来填充Pandas Series y 中的 NaN 值。如果 y 中有 NaN 值，那么会使用线性插值来替换这些 NaN 值。函数返回填充后的 Series y。
def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y

# 该函数输入一个pandas Series类型的时间序列y，如果y的长度超过了限制值limit，
# 就返回按照指定的整数倍数factor进行下采样的时间序列，否则直接返回原序列。函数返回的是一个pandas Series类型的时间序列。
def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
