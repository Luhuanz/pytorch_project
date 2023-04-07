# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Dataset
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm
import logging
import os
import pathlib
import sys
from urllib import request

#从给定的URL中提取文件名，并返回文件名。具体来说，它使用分割符 '/' 将 URL 拆分成多个子字符串，并返回最后一个子字符串，即文件名。
# 如果给定的URL为空，则返回空字符串。
def url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1] if len(url) > 0 else ''


def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """
 #下载进度的函数 progress，其中的参数分别表示已下载的数据块数量、每个数据块的大小和总共要下载的数据大小。
    def progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0##  计算当前下载进度的百分比，
#然后用 sys.stdout.write() 和 sys.stdout.flush() 分别在屏幕上输出下载信息和清除缓存。这个函数在下载数据时被调用，用于实时显示下载进度。
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        sys.stdout.flush()
#从指定的 URL 下载文件并保存到本地文件系统。如果文件已经存在于本地，则不进行下载。
# 如果不存在，则使用 urllib 库中的 urlretrieve 函数下载文件，同时显示下载进度，最后记录下载信息并返回文件路径。
# 如果下载过程中出现错误，将抛出异常。
    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')

#基于 dataclass 的 Python 类定义，定义了一个 M4 数据集类，包含以下成员变量：
@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4Dataset':
       #load用于从缓存中加载数据集。该方法接受两个参数：training表示是否加载训练集，dataset_file表示数据集缓存路径
        info_file = os.path.join(dataset_file, 'M4-info.csv')
        train_cache_file = os.path.join(dataset_file, 'training.npz') #dataset_file是缓存数据集的路径
        test_cache_file = os.path.join(dataset_file, 'test.npz')
        m4_info = pd.read_csv(info_file)
        return M4Dataset(ids=m4_info.M4id.values,
                         groups=m4_info.SP.values,
                         frequencies=m4_info.Frequency.values,
                         horizons=m4_info.Horizon.values,
                         values=np.load(
                             train_cache_file if training else test_cache_file,
                             allow_pickle=True))  # 返回对象
#M4Meta的数据类，其中包含了M4数据集的元信息，如时间序列的季节性、预测长度、历史数据长度等等。
# load_m4_info()函数用于读取M4数据集的信息文件，并返回一个Pandas DataFrame对象，
# 其中包含了所有时间序列的元信息，如ID、季节性、预测长度、历史数据长度、频率等等。
@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # different predict length
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }  # from interpretable.gin


def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)
# M4Dataset 类是用来表示 M4 数据集中的具体时间序列数据，包括其 id、组别、频率、预测长度以及时间序列数据本身等信息，同时还提供了一个方法用于加载数据集。它可以用来创建表示 M4 数据集的具体实例对象。

# M4Meta 类则是用来表示 M4 数据集的元数据，即与数据集有关的一些基本信息，如不同季节性模式的名称、预测长度、历史长度等信息。它提供了一些常量和映射表，用于方便地获取这些信息，同时也可以用来做一些预处理和数据准备工作。