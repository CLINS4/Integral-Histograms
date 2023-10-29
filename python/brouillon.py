import numpy as np

nchannels = 3
nbins = 4
rows = 3
cols = 5

a = np.arange(3*4*3*5).reshape(nchannels, rows, cols, nbins)

print(a)