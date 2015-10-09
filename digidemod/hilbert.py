'Tools for digitally demodulating a signal via the Hilbert transform.'


import numpy as np
from scipy.signal import hilbert

from .sample import decimate


def demodulate(y, Q=1, decimate_kwargs={}):
    '''Demodulate signal `y` via the Hilbert transform.

    Parameters:
    -----------
    y - array_like, (`N`,)
        The signal to be demodulated.
        [y] = arbitrary

    Q - int
        The decimation factor, Q >= 1. For `y` to be demodulated, the fastest
        component in the signal must be the carrier frequency. The features
        to be extracted in the demodulated signal (e.g. the modulating phase)
        vary at rates significantly slower than the carrier frequency.
        To prevent memory waste, then, it is advantageous to decimate
        (i.e. low pass filter and downsample) the demodulated signal.
        [Q] = unitless

    decimate_kwargs - dict
        Keyword arguments for

                :py:function:`decimate <digidemod.sample.decimate>`

        The empty dictionary default will result in the use of the
        default `decimate(...)` keyword arguments.

    Returns:
    --------
    tuple (`t`, `A`, `ph`, `valid`) or (`A`, `ph`, `valid`) where

    t - array_like, ((`N` // Q) + 1,)
        The decimated signal's time base. This will *only* be returned
        if the original signal's time base is provided as a value with
        key 't' in `decimate_kwargs`.
        [tdec] = [decimate_kwargs['t']]

    A - array_like, ((`N` // Q) + 1,)
        The amplitude of the signal `y`.
        [A] = [y]

    ph - array_like, ((`N` // Q) + 1,)
        The phase of the signal `y`. If `y` has *no* phase modulations,
        then `ph` will vary linearly in time with a rate given by
        the carrier frequency of `y`.
        [ph] = radians

    valid - slice
        # The slice corresponding to "valid" indices of `tdec` and `ydec`.
        # The low pass filtering involves a convolution, so boundary
        # effects are visible at the edges of `tdec` and `ydec`.
        # However, boundary effects are *not* present for the slices

        #             `tdec[valid]` and `ydec[valid]`

        # where `valid` corresponds to convolution with mode = 'valid'.

    '''
    # Compute the analytic representation of the signal
    y_a = hilbert(y)

    # Extract the amplitude and phase from the signal
    A = np.abs(y_a)
    ph = np.unwrap(np.angle(y_a))

    # Decimate signal if desired
    if Q > 1:
        if 't' in decimate_kwargs.keys():
            t, A, valid = decimate(A, Q, **decimate_kwargs)
            t, ph, valid = decimate(ph, Q, **decimate_kwargs)
        else:
            A, valid = decimate(A, Q, **decimate_kwargs)
            ph, valid = decimate(ph, Q, **decimate_kwargs)
    elif Q == 1:
        valid = slice(None, None, None)
        if 't' in decimate_kwargs.keys():
            t = decimate_kwargs['t']
    else:
        raise ValueError('`Q` must be an integer >= 1.')

    if 't' in decimate_kwargs.keys():
        return t, A, ph, valid
    else:
        return A, ph, valid
