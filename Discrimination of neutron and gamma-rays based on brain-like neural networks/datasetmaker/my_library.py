import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from library import processing_data as prodata
from scipy.signal import find_peaks
import math

def search_qdc_threshold(energy, data, ToFRange):
    # Find the tagged neutrons up to an energy threshold, rank them by QDC and take the average of the middle 20% as
    # the QDC threshold corresponding to the given energy threshold
    up_E = energy
    low_E = energy - 0.2
    QDC_LG_CH1 = data.query(f'tof_ch2 > {ToFRange[0]} and tof_ch2 < {ToFRange[1]} and tofE_ch2 > {low_E} and tofE_ch2 < {up_E}')['qdc_lg_ch1']
    average_qdc_lg_ch1 = np.mean(np.sort(QDC_LG_CH1)[int(len(QDC_LG_CH1) * 0.4):int(len(QDC_LG_CH1) * 0.6)])
    return average_qdc_lg_ch1

def remove_signal(NE213A_data, group, threshold):
    # Remove the signal from the data
    # 将NE213A_data数据集切分为每一百个信号
    NE213A_data_sorted = NE213A_data.sort_values(by='tofE_ch2', ascending=False)
    chunks = [NE213A_data_sorted[i:i + group] for i in range(0, len(NE213A_data), group)]

    # 定义一个空的DataFrame来存储平均值和方差满足条件的信号
    filtered_dataframe = pd.DataFrame()

    for chunk in chunks:
        # 计算当前一百个信号的qdc通道的平均值和方差
        qdc_mean = chunk['qdc_lg_ch1'].mean()
        qdc_std = chunk['qdc_lg_ch1'].std()

        # 找出偏离平均值Threshold个方差的信号
        deviated_signals = chunk[
            (chunk['qdc_lg_ch1'] >= qdc_mean - qdc_std * threshold) & (chunk['qdc_lg_ch1'] <= qdc_mean + qdc_std * threshold)]

        # 将满足条件的信号加入filtered_dataframe中
        filtered_dataframe = pd.concat([filtered_dataframe, deviated_signals])

    # 重置索引
    filtered_dataframe = filtered_dataframe.reset_index(drop=True)

    return filtered_dataframe


def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) / c) ** 2)


# Plot double Gaussian curve
def plot_double_gaussian(cut_spot, a1, b1, c1, a2, b2, c2, x):
    index = np.abs(x - cut_spot).argmin()
    x1 = x[0:index]
    x2 = x[index:]
    y1 = gaussian(x1, a1, b1, c1)
    y2 = gaussian(x2, a2, b2, c2)
    y3 = np.concatenate([y1, y2])
    plt.plot(x, y3)


def create_data_matrix(df):
    # 获取`samples`字段中最长的行的长度
    max_row_length = max(len(row) for row in df['samples_ch1'])

    # 创建一个全零的数据矩阵"DATA"
    DATA = np.zeros((df.shape[0], max_row_length))

    # 将每一行的数据放入"DATA"矩阵中
    for i, row in enumerate(df['samples_ch1']):
        DATA[i, :len(row)] = row


    return np.array(DATA)

def doubleGaussFunc(x, amp1, mean1, std1, amp2, mean2, std2):
    """
    A double Gaussian function.
    f(x) = G1(x)+G2(x)
    """
    amp1 = np.abs(amp1)
    amp2 = np.abs(amp2)

    return gaussFunc(x, amp1, mean1, std1) + gaussFunc(x, amp2, mean2, std2)

def gaussFunc(x, a, x0, sigma):
    """
        A generic Gaussian function.
        - 'x'......This is the independent variable data (list format).
        - 'a'......This is a constant.
        - 'x0'.....This is the mean value of distribution position.
        - 'sigma'..This is the standard diviation of the distribution.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2019-04-03
    """
    a = np.abs(a)
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def doubleGaussFit(x_input, y_input, start=0, stop=1, param=[1, 1, 1, 1, 1, 1], error=False):
    """
    Fitting routine for doubleGaussFunc() using the scipy.optimize library and curve_fit() routine.

    Parameters:
    x.......x-values for fit.
    y.......y-values for fit.
    param...list of starting parameters to give to fit. Example: param = (amp1, mean1, std1, amp2, mean2, std2)
    start...start values to fit from (x-value)
    stop....stop values to fit to (x-value)
    error...a boolean, indicating if the fitting errors should be returned. Default: False
    """
    # check if all starting paramters are given
    if len(param) != 6:
        print('doubleGaussFit(): All 6 starting parameters needs to be given...')
        return 0

    # check if range is given, else fit entire data-set
    x, y = prodata.binDataSlicer(x_input, y_input, start, stop)

    # fit the data for the specified range
    try:
        popt, pcov = curve_fit(doubleGaussFunc, x, y, p0=param)
    except RuntimeError:
        print('gaussFit(): RuntimeError')
        popt = [1, 1, 1]
        pcov = [1, 1, 1]

    if error:
        return popt, np.sqrt(np.diag(pcov))
    else:
        return popt

def PS_my(data1):
    data = -data1
    n0 = 25
    n1 = 475
    m0 = 35

    LG = 0
    SG = 0
    PS = 0
    max_postion = np.argmax(data)
    LG = np.sum(data[max_postion - n0: max_postion + n1 + 1])
    SG = np.sum(data[max_postion - n0: max_postion + m0 + 1])
    PS = ((LG - SG) / LG)
    return PS

def process_data(data):

    threshold = 0.4
    threshold_2 = 0.3
    m0 = 100
    m1 = 250
    n0 = 2

    data = -data
    threshold = np.max(data) * threshold
    threshold_2 = np.max(data) * threshold_2

    max_idx, _ = find_peaks(data)
    max_val = data[max_idx]

    max_val_sorted = np.sort(max_val)[::-1]
    sort_idx = np.argsort(max_val)[::-1]
    max_idx_sorted = max_idx[sort_idx]
    # max_idx_sorted[0] > m0 and max_idx_sorted[0] < m1 and
    if max_val_sorted[1] < threshold and max_val_sorted[
        2] < threshold_2 and not np.all(np.diff(data[max_idx_sorted[0]: max_idx_sorted[0] + n0]) == 0):
        return True
    else:
        return False

def normalize_signal(signal, range_min=0, range_max=1):
    # 计算信号的最小值和最大值
    min_val = np.min(signal)
    max_val = np.max(signal)

    # 对信号进行归一化
    normalized_signal = (signal - min_val) * (range_max - range_min) / (max_val - min_val) + range_min

    return normalized_signal