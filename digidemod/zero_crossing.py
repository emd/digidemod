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

    def _getRisingZeroCrossingIndices(self):
        '''Get indices immediately preceding a "rising" zero crossing.

        A "rising" zero crossing occurs between two points
        `y_i` and `y_{i + 1}` if and only if

                  y_i < 0     and     y_{i + 1} > = 0

        where "rising" specifies that the function is increasing
        across the zero crossing.

        The motivation for this function is derived from endolith's
        `freq_from_crossings(...)` routine found here:

            https://gist.github.com/endolith/255291

        '''
        return np.where(np.logical_and(self.y[:-1] < 0, self.y[1:] >= 0))[0]

    def _getRisingZeroCrossingTimes(self):
        '''Get times corresponding to a "rising" zero crossing.
        The zero crossing time is determined via linear interpolation
        between the point immediately preceding the zero crossing
        and the point immediately following the zero crossing.

        The motivation for this function is derived from endolith's
        `freq_from_crossings(...)` routine found here:

            https://gist.github.com/endolith/255291

        '''
        # Find all indices immediately preceding a rising zero crossing
        ind = self._getRisingZeroCrossingIndices()

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

    ind = zc._getRisingZeroCrossingIndices()
    plt.plot(t[ind], zc.y[ind], 'sg')

    zc_time = zc._getRisingZeroCrossingTimes()
    plt.plot(zc_time, np.zeros(len(zc_time)), '*r')
