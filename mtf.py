# From: https://gist.github.com/stefanv/2051954

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


N = 100
std = 5#3.75
box = 5

# See also: http://www.imatest.com/docs/sharpness/#calc

def findStepEdgeSubpix(x, r=3):
    """Find the position in x that has highest gradient."""
    x = abs(ndimage.gaussian_filter(x, 2, order=1))
    r = 3
    peak = np.argmax(x)
    p = np.polyfit(np.r_[peak-r:peak+r+1], np.log(x[peak-r:peak+r+1]), 2)
    return -p[1]/(p[0]*2)
    

def superresEdge(edgeProfiles, n=4, returnBins = False):
    """
    Given a bunch of edge profiles, create an average
    profile that is n times the size based on the edge positions.
    """
    edgePositions = np.array(map(findStepEdgeSubpix, edgeProfiles))
    p = np.polyfit(range(len(edgePositions)), edgePositions, 2)
    fitPositions = np.polyval(p, range(len(edgePositions)))
    meanPos = floor(np.mean(fitPositions))
    shifts = (floor(fitPositions) - meanPos)
    bins = np.cast[int](np.modf(fitPositions)[0] * n)
    edgeProfiles = [ndimage.shift(profile, -round(shft), cval=float('nan'), order=0) for profile, shft in zip(edgeProfiles, shifts)]
    toAverage = [[] for i in range(n)]
    for bin_i, profile in zip(bins, edgeProfiles):
        toAverage[bin_i].append(profile)
        
    result = np.zeros(len(edgeProfiles[0]) * n)
    
    for i in range(n):
        if not toAverage[i]: continue
        x = np.mean(toAverage[i], 0)
        #print 'mean',i, x
        result[n-i-1::n] = x # The bins are sorted by increasing order of edge position; we want right-most edge to go in first bin.
    # Trim ends -- shifts were marked with nan.
    result = result[~isnan(result)]
    if returnBins:
        return result, edgeProfiles, toAverage
    return result, edgeProfiles
    


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


if False:
    # Generate edge
    x = np.zeros(N)
    x += 0.1
    x[:N // 2] = 1
    
    # Pass through various filters
    y0 = x[5:-5]
    y1 = ndimage.gaussian_filter1d(x, 3.75)[5:-5]
    y2 = np.convolve(x, 1/box * np.ones(box), mode='same')[5:-5]
    
    Y0 = MTF(y0)
    Y1 = MTF(y1)
    Y2 = MTF(y2)
    
    f, (ax0, ax1) = plt.subplots(1, 2)
    ix = np.arange(len(Y1)) / (2 * len(Y1))
    
    ax0.plot(y0, '.-')
    ax0.plot(y1, '.-')
    ax0.plot(y2, '.-')
    
    
    ax1.plot(ix, np.abs(Y0), '.-', label='Raw')
    ax1.plot(ix, np.abs(Y1), '.-', label='Gaussian %.2f' % std)
    ax1.plot(ix, np.abs(Y2), '.-', label='Box %d' % box)
    ax1.legend()
    
    plt.show()

if __name__ == '__main__':
    x = np.zeros((100,300))
    x[:, x.shape[1]//2:] = 1.0
    x = ndimage.rotate(x, 5, order=1)
	# Mess with the image here:
    # x = ndimage.gaussian_filter(x, 1)
    x = x[x.shape[0]//2-10:x.shape[0]//2+10, x.shape[1]//2-100:x.shape[1]//2+100]
    from pylab import *
    figure()
    n=4
    edges, im, theBins = superresEdge(x, n=n, returnBins=True)
    subplot(2,2,1)
    imshow(x,interpolation='nearest')
    subplot(2,2,3)
    imshow(im, interpolation='nearest')
    subplot(2,2,2)
    plot(edges,'.-')
    subplot(2,2,4)
    Y = MTF(edges/max(edges))
    lpPerPix = n * np.arange(len(Y)) / (2 * len(Y))
    plot(lpPerPix, abs(Y),'.-')
    
    if False:
        figure()
        title('comparing mean of FFT to the ISO 12233 way.')
        lpPerPix_ = np.arange(len(MTF(x[0]))) / (2 * len(MTF(x[0])))
        plot(lpPerPix_,np.mean([np.abs(MTF(xi)) for xi in x],0) / x.max(),'.-');ylim(0,1.0)
        plot(lpPerPix, abs(Y),'g.-')
        
    show()