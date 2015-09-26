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
