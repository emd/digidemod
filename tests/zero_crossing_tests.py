from nose import tools
import numpy as np
from digidemod import ZeroCrossing


def test_constant():
    # Should detect *no* zero crossings with a constant array
    # Specify `AC_coupled` = False to prevent computation of DC offset,
    # which is only a well-defined operation when there are two or more
    # zero crossings of the same type.
    zc = ZeroCrossing(np.array([1, 1]), 1, AC_coupled=False)
    tools.assert_equal(len(zc._getRisingZeroCrossingIndices()), 0)
    tools.assert_equal(len(zc._getRisingZeroCrossingTimes()), 0)


def test_linear():
    # Should detect *one* rising zero crossing
    # Specify `AC_coupled` = False to prevent computation of DC offset,
    # which is only a well-defined operation when there are two or more
    # zero crossings of the same type.
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
    y = np.array([-1., 1., -1., 1., -1., 1.])
    full_cycle_ind = np.array([0, 1, 2, 3])

    zc = ZeroCrossing(y, 1)
    np.testing.assert_equal(full_cycle_ind, zc._getFullCycleIndices())

    # Now, try with the inverse of the above signal
    # such that the first zero crossing is "falling"" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" """ "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "
    y = -y
    zc = ZeroCrossing(y, 1)
    np.testing.assert_equal(full_cycle_ind, zc._getFullCycleIndices())


def test_getDCOffset():
    # Set up a simple triangle wave with 2.5 cycles, with DC offset 0.1
    y_DC = 0.1
    y = y_DC + np.array([-1., 1., -1., 1., -1., 1.])

    zc = ZeroCrossing(y, 1, AC_coupled=False)
    tools.assert_almost_equal(y_DC, zc.getDCOffset())


def test_getRMS():
    # Set up a simple triangle wave with 2.5 cycles
    y = np.array([-1., 1., -1., 1., -1., 1.])

    zc = ZeroCrossing(y, 1)
    tools.assert_equal(1., zc.getRMS())


def test_getZeroCrossingTimes():
    # Set up a simple triangle wave with 2.5 cycles
    y = np.array([-1., 1., -1., 1., -1., 1.])
    t = np.arange(len(y))
    Fs = 1. / np.mean(np.diff(t))
    xtimes = t[:-1] + (0.5 * np.diff(t))

    zc = ZeroCrossing(y, Fs, t0=t[0])
    np.testing.assert_allclose(xtimes, zc.getZeroCrossingTimes())


def test_getFrequency():
    # Create signal with a specified frequency and random initial phase
    f = np.sqrt(2)
    t = np.arange(0, 10, 0.1)
    ph0 = 2 * np.pi * np.random.rand(1)
    y = np.cos((2 * np.pi * f * t) + ph0)

    # Construct zero crossing object and test frequency calculation
    Fs = 1. / np.mean(np.diff(t))
    zc = ZeroCrossing(y, Fs, t0=t[0])
    tools.assert_almost_equal(zc.getFrequency(), f, places=3)


def test_getNumCycles():
    # Create signal with a specified number of cycles
    f = np.sqrt(2)
    t = np.random.rand(1) + np.arange(0, 10, 1. / 11)
    num_cycles = f * (t[-1] - t[0])
    y = np.cos(2 * np.pi * f * t)

    # Construct zero crossing object and test frequency calculation
    Fs = 1. / np.mean(np.diff(t))
    zc = ZeroCrossing(y, Fs, t0=t[0])
    tools.assert_almost_equal(zc.getNumCycles(), num_cycles, places=2)


def test__getRisingZeroCrossingTimes():
    # Create signal with well-known zero crossings
    f = 0.125
    t = 0.5 + np.linspace(0, 100, 101)
    y = np.cos(2 * np.pi * f * t)
    Fs = 1. / np.mean(np.diff(t))

    # Construct zero crossing object
    zc = ZeroCrossing(y, Fs, t0=t[0])

    # Test identification of *rising* zero crossings
    # via linear interpolation
    xtimes_rising_exact = np.arange(6, t[-1], int(1. / f))
    xtimes_rising_calc = zc._getRisingZeroCrossingTimes()
    np.testing.assert_allclose(xtimes_rising_exact, xtimes_rising_calc)

    # Test identification of *falling* zero crossings
    # via linear interpolation
    xtimes_falling_exact = np.arange(2, t[-1], int(1. / f))
    xtimes_falling_calc = zc._getRisingZeroCrossingTimes(invert=True)
    np.testing.assert_allclose(xtimes_falling_exact, xtimes_falling_calc)


def test__getRisingZeroCrossingTimesFit():
    # Create signal with well-known zero crossings
    f = 0.125
    t = 0.5 + np.linspace(0, 100, 101)
    y = np.cos(2 * np.pi * f * t)
    Fs = 1. / np.mean(np.diff(t))

    # Construct zero crossing object
    zc = ZeroCrossing(y, Fs, t0=t[0])

    # Test identification of *rising* zero crossings
    # via fitting a sinusoidal function
    xtimes_rising_exact = np.arange(6, t[-1], int(1. / f))
    xtimes_rising_calc = zc._getRisingZeroCrossingTimesFit()
    np.testing.assert_allclose(xtimes_rising_exact, xtimes_rising_calc)

    # Test identification of *falling* zero crossings
    # via fitting a sinusoidal function
    xtimes_falling_exact = np.arange(2, t[-1], int(1. / f))
    xtimes_falling_calc = zc._getRisingZeroCrossingTimesFit(invert=True)
    np.testing.assert_allclose(xtimes_falling_exact, xtimes_falling_calc)
