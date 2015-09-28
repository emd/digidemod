'Provides a class and functions for demodulating a signal via zero crossing.'


import numpy as np
import matplotlib.pyplot as plt


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
        The digital signal to be demodulated.
        [y] = arbitrary units

    crossing_times - array, (`M`,)
        The times for which signal `y` crosses through zero.
        [crossing_times] = 1 / [Fs]

    '''
    def __init__(self, y, Fs, t0=0, AC_coupled=True, plot=False):
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

        AC_coupled - bool
            If True, remove DC offset from signal `y` before
            computing the zero crossing times

        plot - bool
            If True, plot the original signal, the component
            of the signal used to compute the DC offset/RMS,
            and the computed DC offset.

        '''
        self.Fs = Fs
        self.t0 = t0
        self.y = y

        if AC_coupled:
            self.y -= self.getDCOffset(plot=plot)

        self.crossing_times = self.getZeroCrossingTimes()

        if len(self.crossing_times) > 0:
            self.f = self.getFrequency()

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

    def getZeroCrossingTimes(self):
        'Get times corresponding to signal zero crossings.'
        # Find rising and falling zero crossing times
        rising_xtimes = self._getRisingZeroCrossingTimes()
        falling_xtimes = self._getRisingZeroCrossingTimes(invert=True)

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

    def _getRisingZeroCrossingTimes(self, invert=False):
        '''Get times corresponding to a "rising" zero crossing.
        The zero crossing time is determined via linear interpolation
        between the point immediately preceding the zero crossing
        and the point immediately following the zero crossing.

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

        # Convert the intersample "indices" to times by normalizing
        # to the sample frequency, `Fs`. (Recall:  t_n = n (dt) = n / Fs).
        crossings /= self.Fs

        return crossings


if __name__ == '__main__':
    t0 = 0
    tf = 10
    dt = 0.1
    Fs = 1. / dt

    t = np.arange(t0, tf, dt)
    y = np.cos(2 * np.pi * t)

    zc = ZeroCrossing(y, Fs, t0=t0)

    plt.plot(t, zc.y, '-sb')
    plt.axhline(0, c='k')

    rising_ind = zc._getRisingZeroCrossingIndices()
    plt.plot(t[rising_ind], zc.y[rising_ind], 'sg')

    rising_zc_time = zc._getRisingZeroCrossingTimes()
    plt.plot(rising_zc_time, np.zeros(len(rising_zc_time)), '*r')

    falling_ind = zc._getRisingZeroCrossingIndices(invert=True)
    plt.plot(t[falling_ind], zc.y[falling_ind], 'og')

    falling_zc_time = zc._getRisingZeroCrossingTimes(invert=True)
    plt.plot(falling_zc_time, np.zeros(len(falling_zc_time)), '*r')

    plt.show()
