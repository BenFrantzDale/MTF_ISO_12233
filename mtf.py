from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


N = 100
std = 3.75
box = 5


def MTF(x, window=True):
    """
    Compute the MTF of an edge scan function.

    Parameters
    ----------
    x : 1D ndarray
        Edge scan function.
    window : bool
        Whether to apply Hanning windowing to the input.

    Notes
    -----
    The line spread function is the derivative of the edge scan function.  The
    FFT of the line spread function gives the MTF.

    See Also
    --------
    http://www.cis.rit.edu/research/thesis/bs/2001/perry/thesis.html

    """
    y = np.diff(x)

    if window:
        y = y * np.hanning(len(y))

    y = np.append(y, np.zeros(100))
    Y = np.fft.fft(y)

    return Y[:len(Y) // 2]


# Generate edge
x = np.zeros(N)
x[:N // 2] = 1

# Pass through various filters
y1 = ndimage.gaussian_filter1d(x, 3.75)[5:-5]
y2 = np.convolve(x, 1/box * np.ones(box), mode='same')[5:-5]

Y1 = MTF(y1)
Y2 = MTF(y2)

f, (ax0, ax1) = plt.subplots(1, 2)
ix = np.arange(len(Y1)) / (2 * len(Y1))

ax0.plot(y1)
ax0.plot(y2)

ax1.plot(ix, np.abs(Y1), label='Gaussian %.2f' % std)
ax1.plot(ix, np.abs(Y2), label='Box %d' % box)
ax1.legend()

plt.show()
