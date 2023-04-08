# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#GluonTS 的时间特征模块的基础代码。这个模块提供了一些函数和类，用于从时间戳中提取有用的特征，如小时、分钟、天、月、季度等。
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List

import numpy as np
import pandas as pd
# Pandas 库中处理时间序列的一个核心模块，主要用于创建表示时间偏移量的对象，例如天、小时和分钟等。
from pandas.tseries import offsets
#用于将时间字符串转换为偏移对象。例如，可以将字符串 "D" 转换为 Day 偏移对象，以表示一天的时间偏移量。
# 这个函数也支持其他的时间字符串格式，例如 "H" 表示一个小时的偏移量，"T" 表示一分钟的偏移量等。
from pandas.tseries.frequencies import to_offset
# 例
# dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='MS')
# last_day_of_month = dates + MonthEnd(1)


#这段代码定义了一个TimeFeature类，它是一个基类，它没有实现__call__方法，因此它不能被直接使用，
# 但它可以被子类继承并实现__call__方法，以生成时间特征的向量表示
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

# SecondOfMinute 类是用来计算时间序列中每个时间戳的分钟数所占小时内分钟总数的比例，
#，对于一个时间戳为 2023-04-05 14:25:45 的时间序列，SecondOfMinute 函数会返回一个值为 (45/59)-0.5=-0.22 的一维数组，
# 表示这个时间戳所在秒在这个内占分钟据的比例。
class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
   #例如 频率为每小时一次（hourly），则可以调用以下代码：
   #time_features = time_features_from_frequency_str('1H')
#包含适合每小时一次（hourly）时间序列数据的特征，例如小时、星期几、月份等。可以根据需要将它们组合成一个特征向量来表示时间序列数据的时间特征。
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """
#使用了一些pandas的时间偏移(offset)类，例如offsets.YearEnd、offsets.QuarterEnd、offsets.MonthEnd等，
# 它们表示某一年、季度、月份的末尾。函数根据输入的时间频率字符串判断该字符串对应的时间偏移类型，然后将对应时间偏移类型的时间特征类实例化为列表返回

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
# 当输入'5min'时，函数会返回[MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]，即适合5分钟频率的时间特征类实例列表。
#判断是否输入的时间频率字符串对应该时间偏移类型。如果是，则返回该时间偏移类型对应的时间特征类的实例列表。如果输入的时间频率字符串不属于支持的时间频率，函数会抛出一个运行时错误(RuntimeError)。
    raise RuntimeError(supported_freq_msg)


#time_features函数基于给定的时间序列dates和频率freq，返回该时间序列的时间特征矩阵。
# 其中freq参数指定了时间序列的时间间隔，例如'h'表示每小时，'d'表示每天。
# time_features_from_frequency_str 生成[HourOfDay(),
#  DayOfWeek(), DayOfMonth(),DayOfMonth() 这些对象，然后 time_features 输入一个时间 这个时间以freq值切分，然后分别调用不同的对象
def time_features(dates, freq='h'):

    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

if __name__ == '__main__':
    # print(time_features_from_frequency_str('h')) #[HourOfDay(), DayOfWeek(), DayOfMonth(),DayOfMonth()]
    index = pd.date_range(start='2022-01-01', end='2022-01-02 2:00:00', freq='H')
    # print(time_features_from_frequency_str('h')[0])
    list1=[feat(index) for feat in time_features_from_frequency_str('h')]
#[Float64Index([-0.5, -0.4565217391304348, -0.41304347826086957], dtype='float64'), HourOfDay()
# Float64Index([0.33333333333333337, 0.33333333333333337, 0.33333333333333337], dtype='float64'),  DayOfWeek()
# Float64Index([-0.5, -0.5, -0.5], dtype='float64'),  DayOfMonth()
# Float64Index([-0.5, -0.5, -0.5], dtype='float64')]    DayOfMonth()

    print(list1)