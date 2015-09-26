from nose import tools
import numpy as np
from digidemod import ZeroCrossing


def test_constant():
    # Should detect *no* zero crossings with a constant array
    zc = ZeroCrossing(np.array([1, 1]), 1)
    tools.assert_equal(len(zc._getRisingZeroCrossingIndices()), 0)
    tools.assert_equal(len(zc._getRisingZeroCrossingTimes()), 0)


def test_linear():
    # Should detect *one* rising zero crossing
    zc = ZeroCrossing(np.array([-1, 0, 1]), 1)
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
