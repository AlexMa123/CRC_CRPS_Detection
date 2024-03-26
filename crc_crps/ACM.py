"""
Automated Coordigram Method (ACM)
"""
from .coordigram import get_coordigram, get_selfcoordigram
import numpy as np


def acm_test(rpeak, breath_time, interval_range,
             threshold=0.25):
    """
    Input:
        rpeak: rpeak positions
        breath_time: breath time positions
        interval_range: range of the interval
        threshold: threshold for the test
    Output:
        result: boolean value indicating whether the test is passed
        data_pair: data pair for the ACM test
    """
    coordigram = get_coordigram(rpeak, breath_time, interval_range)
    data_pair = get_selfcoordigram(coordigram)
    data_pair = np.array(data_pair)
    diff = data_pair[:, 0] - data_pair[:, 1]
    width = np.abs(diff).max()
    return width < threshold, data_pair

