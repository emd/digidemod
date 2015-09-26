'Provides a class and functions for demodulating a signal via zero crossing.'


import numpy as np
import matplotlib.pyplot as plt


class ZeroCrossing(object):
    def __init__(self, y, Fs, t0=0):
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

        '''
        self.Fs = Fs
        self.t0 = t0
        self.y = y

    def getZeroCrossings(self):
        pass

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

    def getTimeBase(self):
        'Get time base corresponding to signal `y`.'
        t0 = self.t0
        Fs = self.Fs
        return np.arange(t0, t0 + (len(self.y) / Fs), 1. / Fs)

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
