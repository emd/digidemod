import numpy as np
from digidemod.hilbert import demodulate


def test_demodulate_carrierOnly():
    'Test demodulation on (nearly) perfect carrier wave.'
    # Create linear phase evolution
    # (i.e. the corresponding signal `y` is only the carrier frequency)
    Fs = 200.
    f0 = 27.
    t = np.arange(0, 10, 1. / Fs)
    ph0 = 2 * np.pi * f0 * t

    # Create signal
    A0 = 1.
    y = A0 * np.cos(ph0)

    # Attempt to demodulate
    A, ph, valid = demodulate(y, Q=1, decimate_kwargs={})

    np.testing.assert_allclose(A, A0, atol=1e-11)
    np.testing.assert_allclose(ph, ph0, atol=1e-11)


def test_demodulate_sinusoidalModulation():
    'Test demodulation with a linear phase modulation.'
    # Create linear phase evolution
    # (i.e. the corresponding signal `y` is only the carrier frequency)
    Fs = 200.
    f0 = 27.
    t = np.arange(0, 10, 1. / Fs)
    ph0 = 2 * np.pi * f0 * t

    # Sinusoidal phase modulation
    # Error *decreases* as `fm` and/or `phm` decrease
    fm = 5e-2  # if Fs = 200 MHz, then here fm = 50 kHz
    phm = 1e-2 * np.cos(2 * np.pi * fm * t)

    # Create signal
    A0 = 1.
    y = A0 * np.cos(ph0 + phm)

    # Attempt to demodulate
    # Note: High frequency noise/V-shape of error between
    # calculated phase `ph` and true phase `ph0 + phm`
    # *decreases* as Q is increased (e.g. the low pass filtering
    # in `decimate(...)` is cleaning up the signal/removing errors).
    Q = 20
    A, ph, valid = demodulate(y, Q=Q, decimate_kwargs={})

    np.testing.assert_allclose(A[valid], A0, atol=1e-5)
    np.testing.assert_allclose(ph[valid], (ph0 + phm)[::Q][valid], atol=1e-5)
