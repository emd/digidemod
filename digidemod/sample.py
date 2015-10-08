'Provides routines for (re)sampling and or decimating a signal.'


import numpy as np
from filters.fir import NER


def decimate(y, Q):
    '''Decimate signal `y` by integer factor `Q`.

    Parameters:
    -----------
    y - array_like, (`N`,)
        The signal to be decimated.
        [y] = arbitrary

    Q - int
        The decimation factor. Q must be a positive integer;
        a ValueError is raised otherwise.

    Returns:
    --------
    ydec - array_like, (`N`)
        The decimated signal.
        [ydec] = [y]

    '''
    if (Q < 1) or not isinstance(Q, int):
        raise ValueError('`Q` must be a positive integer.')

    pass
