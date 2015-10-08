from nose import tools
import numpy as np
from digidemod.sample import decimate


def test_decimate_valueErrors():
    # Attempts to decimate by a *negative* value should fail
    tools.assert_raises(ValueError, decimate, np.array([1]), -1)

    # Attempts to decimate by anything other than an int should fail
    tools.assert_raises(ValueError, decimate, np.array([1]), 1.5)

    # The transition band of the low pass filter cannot occupy
    # a negative fraction of the filtered signal's spectral bandwidth
    tools.assert_raises(ValueError, decimate,
                        np.array([1]), 10, {'trans': -0.1})

    # The transition band of the low pass filter cannot occupy
    # a bandwidth larger than that of the filtered signal's bandwidth
    tools.assert_raises(ValueError, decimate,
                        np.array([1]), 10, {'trans': 1.1})

    # (Optional) time base and signal must have same length
    tools.assert_raises(ValueError, decimate,
                        np.zeros(10), 10, {'t': np.zeros(9)})
