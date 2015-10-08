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
    #
    # Note: Need to specify `t` in **kwargs here (with the default
    # argument), otherwise running `nosetests` results in
    # TypeError: unhashable type (?)
    tools.assert_raises(ValueError, decimate,
                        np.array([1]), 10, {'t': None, 'trans': -0.1})

    # The transition band of the low pass filter cannot occupy
    # a bandwidth larger than that of the filtered signal's bandwidth
    #
    # Note: Need to specify `t` in **kwargs here (with the default
    # argument), otherwise running `nosetests` results in
    # TypeError: unhashable type (?)
    tools.assert_raises(ValueError, decimate,
                        np.array([1]), 10, {'t': None, 'trans': 1.1})

    # (Optional) time base and signal must have same length
    tools.assert_raises(ValueError, decimate,
                        np.zeros(100), 10, {'t': np.zeros(99)})


def test_decimate():
    # Construct deliberately "oversampled" signal
    Fs = 200
    t = np.arange(0, 10, 1. / Fs)
    f0 = 1 + np.random.rand()
    y0 = np.cos(2 * np.pi * f0 * t)

    # Add in Gaussian noise
    y = y0 + (0.1 * (np.random.rand(len(y0)) - np.random.rand(len(y0))))

    # Decimate by factor `Q`
    Q = 20
    tdec, ydec, valid = decimate(y, Q, t=t)

    # Ensure `ydec` is "close" to true signal `y0'
    np.testing.assert_almost_equal(ydec[valid], y0[::Q][valid], decimal=1)

    # ... and make sure `ydec` is *less* noisy than `y`
    ydec_rms = np.std(ydec[valid] - y0[::Q][valid])
    y0_rms = np.std(y[::Q][valid] - y0[::Q][valid])
    tools.assert_less(ydec_rms, y0_rms)
