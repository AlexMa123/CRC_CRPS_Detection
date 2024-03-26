
import numpy as np
from numba import njit


@njit
def get_coordigram(rpeak, breath_time, interval_range=(-3, 0)):
    """
    Compute coordigram based on rposition and breathing setup

    Parameters:
    rpeak: list of rpeak positions (in seconds)
    breath_time: list of breathing positions (in seconds)
    interval_range: range of the interval

    return:
    List[np.array(np.float)]
    """
    coordigram = []
    start_i = 0
    end_i = 1
    for bt in breath_time:
        while start_i < rpeak.size and rpeak[start_i] < bt + interval_range[0]:
            start_i += 1
        end_i = start_i
        while end_i < rpeak.size and rpeak[end_i] <= bt + interval_range[1]:
            end_i += 1
        coordigram.append(rpeak[start_i:end_i] - bt)
    return coordigram


@njit
def get_selfcoordigram(coordigram):
    """
    Compute the data pairs for applying the ACM method
    """
    data_pair = []
    for i in range(len(coordigram) - 1):
        hr1 = coordigram[i]
        hr2 = coordigram[i + 1]

        if hr1.size <= hr2.size:
            hr_smaller = hr1
            hr_larger = hr2
            inverse = True
        else:
            hr_smaller = hr2
            hr_larger = hr1
            inverse = False
        num_points = hr_smaller.size

        for j in range(num_points):
            hr2_point = hr_smaller[j]
            idx = np.abs(hr_larger - hr2_point).argmin()
            hr1_point = hr_larger[idx]
            if inverse:
                hr1_point, hr2_point = hr2_point, hr1_point
            data_pair.append((hr1_point, hr2_point))
    return data_pair
