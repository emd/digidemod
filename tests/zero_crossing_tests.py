from nose import tools
import numpy as np
from digidemod import ZeroCrossing


def test_constant():
    # Should detect *no* zero crossings with a constant array
    zc = ZeroCrossing(np.array([1, 1]), 1, AC_coupled=False)
    tools.assert_equal(len(zc._getRisingZeroCrossingIndices()), 0)
    tools.assert_equal(len(zc._getRisingZeroCrossingTimes()), 0)


def test_linear():
    # Should detect *one* rising zero crossing
    zc = ZeroCrossing(np.array([-1, 0, 1]), 1, AC_coupled=False)
    tools.assert_equal(zc._getRisingZeroCrossingIndices(), np.array([0]))
    tools.assert_equal(zc._getRisingZeroCrossingTimes(), np.array([1]))


def test_getTimeBase():
    # Set up a typical signal and time base
    Fs = 10.
    t0 = 0
    tf = 10
    t = np.arange(t0, tf, 1. / Fs)
    y = np.cos(2 * np.pi * t)

    # Ensure that constructed time base is equivalent to `t`
    zc = ZeroCrossing(y, Fs, t0=t0)
    np.testing.assert_allclose(t, zc.getTimeBase())


def test_getFullCycleIndices():
    # Set up a simple triangle wave with 2.5 cycles.
    # Note that the first zero crossing is a rising zero crossing.
    y = np.array([-1, 1, -1, 1, -1, 1])
    full_cycle_ind = np.array([0, 1, 2, 3])

    zc = ZeroCrossing(y, 1)
    np.testing.assert_equal(full_cycle_ind, zc._getFullCycleIndices())

    # Now, try with the inverse of the above signal
    # such that the first zero crossing is "falling"
    y = -y
    zc = ZeroCrossing(y, 1)
    np.testing.assert_equal(full_cycle_ind, zc._getFullCycleIndices())
