'Provides routines for (re)sampling and or decimating a signal.'


from filters.fir import NER


def decimate(y, Q, t=None, trans=0.5, ripple=0.02):
    '''Decimate signal `y` by integer factor `Q`.

    The decimation is performed by passing the signal `y` through a
    Nearly Equal Ripple (NER) low pass filter and then slicing.
    NER is an FIR filtering technique that allows the user to specify
    the maximum deviation in the passband and stopband ripple. Further,
    the NER filter employed here uses an *odd* number of taps;
    it is a general result that an FIR filter with an odd number of taps
    is "zero phase" - that is, it introduces *no* phase delay between
    the input signal and the resulting filtered signal.

    The -3 dB cutoff frequency of the NER low pass filter is specified
    such that the Nyquist frequency of the decimated signal lies
    at the boundary between the transition band and stopband.

    Parameters:
    -----------
    y - array_like, (`N`,)
        The signal to be decimated.
        [y] = arbitrary

    Q - int
        The decimation factor. Q must be a positive integer;
        a ValueError is raised otherwise.

    t - array_like, (`N`,)
        The time base corresponding to input signal `y`.
        If `t` is provided, a new time base corresponding
        to the decimated signal will additionally be returned.
        [t] = arbitrary

    trans - float
        The fraction of the decimated signal's spectrum in the
        transition band. To ensure a finite passband, 0 < trans < 1,
        with values outside of this bound raising a ValueError.
        [trans] = unitless

    ripple - float
        The maximum deviation in the passband and stopband ripple.
        [ripple] = unitless

    Returns:
    --------
    Tuple (`tdec`, `ydec`, `valid`) or (`ydec`, `valid`) where

    tdec - array_like, ((`N` // Q) + 1,)
        The decimated signal's time base. This will *only* be returned
        if the original signal's time base is provided in the function call.
        [tdec] = [t]

    ydec - array_like, ((`N` // Q) + 1,)
        The decimated signal.
        [ydec] = [y]

    valid - slice
        The slice corresponding to "valid" indices of `tdec` and `ydec`.
        The low pass filtering involves a convolution, so boundary
        effects are visible at the edges of `tdec` and `ydec`.
        However, boundary effects are *not* present for the slices

                    `tdec[valid]` and `ydec[valid]`

        where `valid` corresponds to convolution with mode = 'valid'.

    '''
    if (Q < 1) or not isinstance(Q, int):
        raise ValueError('`Q` must be a positive integer.')

    if (trans < 0) or (trans > 1):
        raise ValueError('`trans` must be between 0 and 1.')

    if (t is not None) and (len(t) != len(y)):
        raise ValueError('`t` must have same dimensions as `y`.')

    # We require that the Nyquist frequency of the *decimated* signal f'_Ny
    # is at the boundary between the filter's transition band and stopband,
    # from which we can easily derive the following formula for the
    # filter's -3 dB cutoff frequency.
    fc = 1 - (0.5 * trans)

    # Create low pass filter. Note that `fc` and `trans` are normalized
    # to the Nyquist frequency f'_Ny of the decimated signal, and that the
    # sampling frequency Fs of the original signal `y` can be expressed as
    #
    #                       Fs / f'_Ny = 2 * Q
    #
    lpf = NER(ripple, [fc], trans, 2 * Q, pass_zero=True)

    # Filter and decimate signal
    ydec = lpf.applyTo(y)[::Q]

    # However, boundary effects will be visible in the edges of `ydec`.
    # Let's find the corresponding "valid" indices where boundary effects
    # are *not* present.
    valid = lpf.getValidSlice()
    valid.start = (valid.start // Q) + 1
    valid.stop = (valid.stop // Q) - 1

    if t is None:
        return ydec, valid
    else:
        return t[::Q], ydec, valid
