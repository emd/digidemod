'Provides a class and functions for demodulating a signal via zero crossing.'


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class ZeroCrossing(object):
    '''Zero crossing object.

    Attributes:
    -----------
    Fs - float
        Digitization rate for signal `y`.
        [Fs] = samples / [time], where [time] = any convenient unit

    t0 - float
        The initial time corresponding to `y[0]`.
        [t0] = 1 / [Fs]

    y - array, (`N`,)
        The digital (sinusoidal) signal from which the zero crossings
        will be found.
        [y] = arbitrary units

    rms - float
        The RMS of signal `y`.
        [rms] = [y]

    crossing_times - array, (`M`,)
        The times for which signal `y` is calculated to cross through zero.
        [crossing_times] = 1 / [Fs]

    f - float
        The calculated frequency of (sinusoidal) signal `y`.
        [f] = [Fs]

    '''
    def __init__(self, y, Fs, t0=0, auto=True,
                 mode='lerp', Npts=4, AC_coupled=True):
        '''Initialize zero crossing demodulator object.

        Parameters:
        -----------
        y - array, (`N`,)
            The digital signal to be demodulated.
            [y] = arbitrary units

        Fs - float
            Digitization rate for signal `y`.
            [Fs] = samples / [time], where [time] = any convenient unit

        t0 - float
            The initial time corresponding to `y[0]`.
            [t0] = 1 / [Fs]

        auto - bool
            If True, automatically process data.

        mode - string
            The method by which the zero crossings are calculated.
            Allowable options are:

                'lerp': linear interpolation between the two
                        data points closest to the zero crossing

                'fit':  fitting a general sinusoidal function to the
                        `Npts` data points closest to the zero crossing

        Npts - int
            The number of data points surrounding a zero crossing
            to be included if `mode` is specified as 'fit'.
            If `Npts` is not even, it is always rounded to
            the next lowest even integer. This ensures that
            there are equal numbers of points both above
            and below the zero crossing. If `mode` is specified
            as 'lerp', `Npts` has no effect.

        AC_coupled - bool
            If True, remove DC offset from signal `y` before
            computing the zero crossing times

        '''
        self.Fs = Fs
        self.t0 = t0
        self.y = y.copy()

        if auto:
            self.process(mode=mode, Npts=Npts, AC_coupled=AC_coupled)

    def process(self, mode='lerp', Npts=4, AC_coupled=True):
        'Perform typical zero crossing calculations.'
        if mode not in set(['lerp', 'fit']):
            raise ValueError("Mode may only be 'lerp' or 'fit'.")

        if AC_coupled:
            self.y -= self.getDCOffset()

        self.rms = self.getRMS()

        # If `mode` is specified as 'fit', the sinusoidal fitting process
        # still relies on estimates from the (cruder) linear interpolation,
        # so we will always perform 'lerp' detection
        self.crossing_times = self.getZeroCrossingTimes(mode='lerp')

        self.f = self.getFrequency()

        if mode is 'fit':
            self.crossing_times = self.getZeroCrossingTimes(
                mode='fit', Npts=Npts)
            self.f = self.getFrequency()

        return

    def getTimeBase(self):
        'Get time base corresponding to signal `y`.'
        t0 = self.t0
        Fs = self.Fs
        return np.arange(t0, t0 + (len(self.y) / Fs), 1. / Fs)

    def getDCOffset(self, plot=False):
        '''Get signal mean over an integer number of cycles.
        (Note that non-integer/fractional cycles will bias the estimate.)

        '''
        # Compute mean over all of the full cycles in signal
        ind = self._getFullCycleIndices()
        offset = np.mean(self.y[ind])

        if plot:
            plt.figure()
            t = self.getTimeBase()
            plt.plot(t, self.y, '-sb')
            plt.plot(t[ind], self.y[ind], '-sr')
            plt.axhline(offset, c='k')

            plt.xlabel('t')
            plt.ylabel('y')
            plt.legend(['signal', 'full cycles', 'computed mean'],
                       loc='lower right')

            plt.show()

        return offset

    def getRMS(self):
        'Get RMS over all of the full cycles in signal.'
        ind = self._getFullCycleIndices()
        return np.std(self.y[ind])

    def getZeroCrossingTimes(self, mode='lerp', Npts=4):
        'Get times corresponding to signal zero crossings.'
        # Find rising and falling zero crossing times
        if mode is 'lerp':
            rising_xtimes = self._getRisingZeroCrossingTimesLerp()
            falling_xtimes = self._getRisingZeroCrossingTimesLerp(invert=True)
        elif mode is 'fit':
            rising_xtimes, exclusions = self._getRisingZeroCrossingTimesFit(
                Npts=Npts)
            falling_xtimes, exclusions = self._getRisingZeroCrossingTimesFit(
                Npts=Npts, invert=True)
        else:
            raise ValueError("Mode may only be 'lerp' or 'fit'.")

        # Merge and sort the zero crossing arrays
        xtimes = np.concatenate((rising_xtimes, falling_xtimes))
        xtimes.sort()

        return xtimes

    def getFrequency(self):
        'Get frequency (units = [self.Fs]) of signal `self.y`.'
        # Get slope of line fit to the zero crossing data
        # [m] = 1 / [self.Fs]
        m = self._fit()[0]

        # Adjacent zero crossings are spaced by *half* a period;
        # that is, T = 2 * m, where T = 1 / f and m = slope from above
        # [f] = [self.Fs]
        f = 0.5 / m

        return f

    def getNumCycles(self):
        '''Get number of cycles in signal `self.y`. Additionally, multiplying
        the cycles by 2 * pi converts to radians.

        '''
        # Get number of cycles corresponding to zero crossings,
        # noting that each successive pair of zero crossings corresponds
        # to an additional half-cycle
        N = 0.5 * (len(self.crossing_times) - 1)

        # However, we still need to account for the fractional cycles
        # before the first zero crossing and after the last zero crossing
        t = self.getTimeBase()

        # Time window between first and last *digitized* point
        T = t[-1] - t[0]

        # Time window between first and last *zero crossing*
        Tzc = self.crossing_times[-1] - self.crossing_times[0]

        N += self.f * (T - Tzc)

        return N

    def _fit(self):
        'Fit zero crossings to a line return relevant fitting parameters.'
        ind = np.arange(len(self.crossing_times))
        return np.polyfit(ind, self.crossing_times, 1)

    def _getFullCycleIndices(self):
        '''Get indices corresponding to "full"/complete cycles in signal,
        where a full cycle is defined as the signal between two successive
        rising (or falling) zero crossings. The particular convention adopted
        is determined by which type of zero crossing occurs the earliest;
        that is, if the earliest zero crossing is rising, a cycle is defined
        as the points between two successive rising zero crossings.

        '''
        rising_ind = self._getRisingZeroCrossingIndices()
        falling_ind = self._getRisingZeroCrossingIndices(invert=True)

        if rising_ind[0] < falling_ind[0]:
            # Cycle defined by two successive rising zero crossings
            crossing_ind = self._getRisingZeroCrossingIndices()
        else:
            # Cycle defined by two successive falling zero crossings
            crossing_ind = self._getRisingZeroCrossingIndices(invert=True)

        return np.arange(crossing_ind[0], crossing_ind[-1], 1)

    def _getRisingZeroCrossingIndices(self, invert=False):
        '''Get indices immediately preceding a "rising" zero crossing.

        A "rising" zero crossing occurs between two points
        `y_i` and `y_{i + 1}` if and only if

                  y_i < 0     and     y_{i + 1} > = 0

        where "rising" specifies that the function is increasing
        across the zero crossing.

        The motivation for this function is derived from endolith's
        `freq_from_crossings(...)` routine found here:

            https://gist.github.com/endolith/255291

        Parameters:
        -----------
        invert - bool
            If True, find "falling" zero crossings as opposed to rising.
            Functionally, this is simply accomplished by finding the
            rising zeros of the *negative* of the original function.

        '''
        if invert:
            y = -self.y
        else:
            y = self.y

        return np.where(np.logical_and(y[:-1] < 0, y[1:] >= 0))[0]

    def _getRisingZeroCrossingTimesLerp(self, invert=False):
        '''Get times corresponding to a "rising" zero crossing.

        The zero crossing time is determined via linear interpolation (lerp)
        between the point immediately preceding the zero crossing and
        the point immediately following the zero crossing.

        The motivation for this function is derived from endolith's
        `freq_from_crossings(...)` routine found here:

            https://gist.github.com/endolith/255291

        Parameters:
        -----------
        invert - bool
            If True, find "falling" zero crossings as opposed to rising.
            Functionally, this is simply accomplished by finding the
            rising zeros of the *negative* of the original function.

        '''
        # Find all indices immediately preceding a rising zero crossing
        ind = self._getRisingZeroCrossingIndices(invert=invert)

        # Find intersample zero-crossing "indices" using linear interpolation
        crossings = np.asarray(
            [i - (self.y[i] / (self.y[i + 1] - self.y[i])) for i in ind])

        # Convert the intersample "indices" to relative times by normalizing
        # to the sample frequency, `Fs`. (Recall:  t_n = n (dt) = n / Fs).
        crossings /= self.Fs

        # Convert relative times to absolute times
        crossings += self.t0

        return crossings

    def _getRisingZeroCrossingTimesFit(self, Npts=4, invert=False):
        '''Get times corresponding to a "rising" zero crossing.

        The zero crossing time is determined via fitting to a
        general sinusoidal function.

        Parameters:
        -----------
        Npts - int
            Number of points about zero crossing to include in fit.
            To ensure only an equal number of points are on each
            side of the zero crossing, `Npts` is automatically rounded
            to the next lowest even integer. If there are ~10 points
            per cycle, then `Npts` = 4 appears to give good results.

        invert - bool
            If True, find "falling" zero crossings as opposed to rising.
            Functionally, this is simply accomplished by finding the
            rising zeros of the *negative* of the original function.

        Returns:
        --------
        tuple - (crossings, exclusions), where:

        crossings - array, (`N`,)
            The zero crossing times detected via fitting to a sine function.
            [crossings] = 1 / [self.Fs]

        exclusions - list
            Occasionally, there will be insufficient data surrounding
            particular zero crossings, and the attempts to fit to a
            sinusoidal function will fail. If this occurs, these
            zero crossing are excluded from the dataset and their
            index (relative to the other zero crossings in `self.y`)
            are returned in `exclusions`.

        '''
        # Only fit an *even* number of points such that
        # there is an equal number of points on each side
        # of the zero crossing
        print 'Using Npts = %i per fit.' % (2 * (Npts / 2))

        # Find all indices immediately preceding a rising zero crossing
        ind = self._getRisingZeroCrossingIndices(invert=invert)

        # Rising zero crossing times obtained from linear interpolation
        # between two nearest points to zero crossing
        crossings = self._getRisingZeroCrossingTimesLerp(invert=invert)

        # Some zero crossings (at the beginning or end of the signal)
        # may have insufficient surrounding data points to perform fit.
        # While we could simply use the zero crossing estimate
        # from linear interpolation for such points, this can
        # significantly decrease the accuracy of subsequent calculations
        # (e.g. signal frequency estimation). Thus, the `exclusions` list
        # will accumulate indices of such points so that they can later
        # be deleted from the dataset.
        exclusions = []

        # For each zero crossing, create a slice of data (of size `Npts`)
        # about the zero crossing and fit to a sinusoid. The value of
        # the zero crossing can then be extracted from the fit.
        for i in np.arange(len(crossings)):
            # Determine index of first point to include in fit.
            # An additional value of `1` is subtracted because
            # `ind[i]` immediately *precedes* the zero crossing
            ind0 = ind[i] - (Npts / 2) - 1

            # ... but don't allow an illegal index. In practice,
            # this is typically only a (potential) problem for
            # the first zero crossing.
            if ind0 < 0:
                ind0 = 0

            # Similarly, determine index of last point to include in fit
            indf = ind[i] + (Npts / 2)

            # ... but again, don't allow an illegal index. In practice,
            # this is typically only a (potential) problem for
            # the last zero crossing.
            if indf > len(self.y):
                indf = len(self.y)

            # Slice original signal to obtain data points for fitting
            t = self.getTimeBase()[ind0:indf]
            y = np.copy(self.y[ind0:indf])

            if invert:
                y *= -1

            # Collect estimated fitting parameters
            p0 = np.array([np.sqrt(2) * self.rms, self.f, crossings[i]])

            # Fit data and extract zero crossing time
            try:
                p = curve_fit(self._sinusoid, t, y, p0=p0)[0]
                crossings[i] = p[-1]

            except TypeError:
                print 'Insufficient surrounding data to perform fit at:'
                print '    zero crossing # = %i' % i
                print '    t = %f' % t[ind[i]]
                print '    ind0 = %i' % ind0
                print '    indf = %i' % indf
                print 'Excluding this zero crossing from the data set.'

        # Remove zero crossings that had insufficient
        # surrounding data points to perform the fit
        crossings = np.delete(crossings, exclusions)

        return crossings, exclusions

    def _sinusoid(self, t, A, f0, t0):
        'General form of a sinusoidal function for curve fitting purposes.'
        return A * np.sin(2 * np.pi * f0 * (t - t0))
