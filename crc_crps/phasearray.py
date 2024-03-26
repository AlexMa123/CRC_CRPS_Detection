import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import matplotlib.pyplot as plt
from numba import vectorize, njit

HANDLED_FUNCTION = {}


@njit
def get_stats(phase, n):
    """
    get mean. std. and var of the phases
    """
    imin = 0
    stdmin = 0
    phase = phase / n
    sorted_phase = np.sort(phase)
    if phase.size < 10:
        step = 1
    else:
        step = int(phase.size / 10)
    for i, p in enumerate(sorted_phase[::step]):
        std_value = np.std(change_delta_phase(phase - p))
        if i == 0:
            stdmin = std_value
        else:
            if stdmin > std_value:
                imin = i
                stdmin = std_value
    mean_value = np.mean(change_delta_phase(
        phase - sorted_phase[imin * step])) + sorted_phase[imin * step]
    mean_value = mean_value % (np.pi * 2)
    return mean_value * n, stdmin * n, stdmin ** 2 * n ** 2


@njit
def remove_mean(phase, n=1, mean_value=-1):
    if mean_value == -1:
        mean_vector = np.cos(phase).mean() + np.sin(phase).mean() * 1j
        mean_value = np.angle(mean_vector)
    result = phase.copy()
    for i, p in enumerate(phase):
        result[i] = result[i] - mean_value
        if result[i] > np.pi * n:
            result[i] = result[i] - 2 * n * np.pi
        if result[i] < - np.pi * n:
            result[i] = result[i] + 2 * n * np.pi
    return result


@njit
def remove_first_phase(phase, n=1):
    result = phase - phase[0]
    for i, p in enumerate(result):
        if result[i] > np.pi * n:
            result[i] = result[i] - 2 * n * np.pi
        if result[i] < - np.pi * n:
            result[i] = result[i] + 2 * n * np.pi
    return result


@vectorize
def change_delta_phase(delta_phase):
    if delta_phase > np.pi:
        return delta_phase - np.pi * 2
    elif delta_phase < - np.pi:
        return delta_phase + np.pi * 2
    else:
        return delta_phase


class PhaseArray(NDArrayOperatorsMixin):
    def __init__(self, data, n=1):
        self._class_name = 'PhaseArray'
        if not isinstance(data, np.ndarray):
            self.data = np.array(data) % (np.pi * 2 * n)
        else:
            self.data = data % (np.pi * 2 * n)
        self.n = n
        self._method = ""
        self.range = (0, np.pi * 2)

    def __instancecheck__(self, instance) -> bool:
        return hasattr(self, '_class_name') and self._class_name == 'PhaseArray'

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.data.shape

    def syn_index(self, method='fourier', **options):
        if method == 'fourier':
            if hasattr(self, '_psi_f'):
                return self._psi_f
            psi = np.sqrt(np.cos(self.data).mean() ** 2 +
                          np.sin(self.data).mean() ** 2)
            self._psi_f = psi
        elif method == 'entropy':
            if hasattr(self, '_psi_e'):
                return self._psi_e
            nbins = options.get('N', default=20)
            hist, _ = np.histogram(self.data, bins=nbins, range=self.range)
            hist = hist / hist.sum()
            S_max = np.log(nbins)
            S = - np.sum(hist * np.log(hist))
            psi = (S_max - S) / S_max
            self._psi_e = psi
        else:
            raise NotImplementedError
        return psi

    def plot_phase(self, plot_mean=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(self.data / self.n, np.ones_like(self.data), label="data")
        if plot_mean:
            mean_phase = self.mean() / self.n
            ax.scatter(mean_phase, 1.4, color='r',
                       label=f"$\\theta$={mean_phase/np.pi/2 * 360 * self.n:.2f}")
            ax.arrow(0, 0, mean_phase, 1.4, color='r')
        ax.set_rticks([1])
        ax.set_thetagrids(90 * np.arange(4),
                          [f'{self.n * i / 2} $\pi$' for i in range(4)])
        ax.set_rlim(0, 1.5)
        ax.legend()
        return fig, ax

    def remove_mean(self):
        # mean_value = self.mean()
        removed_data = remove_mean(
            self.data, self.n) + np.pi * self.n
        self._width = removed_data.max() - removed_data.min()
        return PhaseArray(removed_data, self.n)

    def start_from_zero(self):
        removed_data = remove_first_phase(self.data, self.n) + np.pi * self.n
        return PhaseArray(removed_data, self.n)

    def width(self):
        if not hasattr(self, '_width'):
            self.remove_mean()
        return self._width

    def mean(self):
        if not hasattr(self, 'mean_value'):
            self.mean_value, self.std_value, self.var_value = get_stats(
                self.data, self.n)
        return self.mean_value

    def std(self):
        if not hasattr(self, 'std_value'):
            _ = self.mean()
        return self.std_value

    def var(self):
        if not hasattr(self, 'var_value'):
            _ = self.mean()
        return self.var_value

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.data.shape}, data={self.data})"

    def __sub__(self, other):
        result = np.subtract(self, other)
        result.data = change_delta_phase(result.data)
        return result

    def __getitem__(self, key):
        return self.__class__(self.data[key])

    def __setitem__(self, key, val):
        self.data[key] = val

    def __array__(self, dtype=None):
        if dtype is not None:
            return self.data.astype(float)
        return self.data

    def __array_ufunc__(self, ufunc, method, *input, **kwargs):
        if method == '__call__':
            input = [i.data if isinstance(
                i, self.__class__) else i for i in input]
            result = self.__class__(ufunc(*input, **kwargs))
            if isinstance(result, np.ndarray):
                if np.issubdtype(result.dtype, np.number):
                    return self.__class__(result)
                else:
                    return result
            else:
                return result
        else:
            raise NotImplementedError

    def __array_function__(self, func, types, args, kwargs):
        args = [i.data if isinstance(i, self.__class__) else i for i in args]
        return func(*args, **kwargs)
