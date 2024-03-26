"""
Reduced Synchronization Measure (RSM)    
"""

import numpy as np
from scipy.stats import ttest_1samp
from .phasearray import PhaseArray


def RMStest(phases: PhaseArray, n_group, threshold):
    """
    Input:
        phases: phase points $\Phi^m(t)$
        n_group: number of groups
        threshold: threshold for the test
        
    Output:
        result: boolean value indicating whether the test is passed
        pvalue: p-value of the ttest test 
            (weather the mean of the collapsed phases is 0)
        width: width of the collapsed phases
        collapsed_phases: collapsed phases
    """
    if not isinstance(phases, PhaseArray):
        phase_range = np.max(phases) - np.min(phases)
        n = np.ceil(phase_range / (2 * np.pi))
        phases = PhaseArray(phases, phase_range)
    else:
        n = phases.n

    collapsed_phases = PhaseArray(np.zeros(phases.shape), int(n))
    for i in range(n_group):
        collapsed_phases[i::n_group] = phase_range[i::n_group].remove_mean()

    width = np.max(collapsed_phases) - np.min(collapsed_phases)

    _, pvalue = ttest_1samp(np.array(collapsed_phases), 0)

    return (width < threshold), pvalue, width, collapsed_phases
