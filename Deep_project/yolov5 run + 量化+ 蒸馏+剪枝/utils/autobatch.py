# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile

#检查YOLOv5的训练批量大小，并计算最佳批量大小。
def check_train_batch_size(model, imgsz=640, amp=True):
#在函数内部，它通过使用torch.cuda.amp.autocast来检查当前的精度模式，然后使用autobatch函数来计算最佳的批量大小，最后返回计算得到的批量大小。
    # Check YOLOv5 training batch size
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size


def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    # Automatically estimate best YOLOv5 batch size to use `fraction` of available CUDA memory
    # Usage:
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))

    # Check device
    prefix = colorstr('AutoBatch: ')
#prefix 是一个字符串，包含在 colorstr() 函数中，该函数将字符串的前缀设置为 ANSI 颜色代码，以使输出具有不同的颜色。在
    LOGGER.info(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size
#该语句判断当前是否开启了cudnn.benchmark，如果开启了，则会自动寻找最优的卷积实现算法
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f'{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}')
        return batch_size

    # Inspect CUDA memory
#这行代码的作用是定义一个变量 gb，它表示的是 2 的 30 次方，即 1GB 的大小，用于方便地进行内存大小的计算。
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # 获取指定设备device的属性信息。s
    t = properties.total_memory / gb  # 将该设备的总内存大小除以1024 1024 1024得到总内存大小（以GiB为单位）。
    r = torch.cuda.memory_reserved(device) / gb  # 使用torch.cuda.memory_reserved(device)获取已保留的内存大小
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free 计算可用内存大小，即总内存减去已保留和已分配内存大小。
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')

    # Profile batch sizes
# 获取当前设备的一些基本属性信息，包括总显存大小、已分配显存大小和剩余显存大小，以便后续计算。然后，定义了一些待测试的batch sizes（1、2、4、8和16）
  #通过循环遍历待测试的batch sizes，分别对模型进行3次前向推理，并计算平均推理时间，以此来评估模型在该batch size下的最大承受能力。最终，该函数会返回模型在每个测试batch size下的平均推理时间，以及一个建议的最大批量大小。
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')
#自动计算适合当前GPU的YOLOv5模型的最佳batch size。
# 它首先通过检查当前GPU的内存使用情况来确定可用的内存量，然后通过测试一系列预定义的batch size来找到最佳的batch size，
    # 以便在保证训练速度的同时尽可能利用可用的内存。最终它会输出最佳的batch size，并将其用于YOLOv5模型的训练。
    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size
        LOGGER.warning(f'{prefix}WARNING ⚠️ CUDA anomaly detected, recommend restart environment and retry command.')

    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    LOGGER.info(f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅')
    return b
