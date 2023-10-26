import cv2 as cv
import numpy as np


def construct_IntegralHistogram(dim, nchannels, nbins): #dim = (height,width)
    return (dim[0],dim[1],nchannels,nbins)


def integralHistogram(IH,im): #IH = (height, width, nchannels, nbins)
    rows,cols, nchannels,nbins = IH
    hist_rows = rows +1
    hist_cols = cols +1
    hist_len = hist_rows * hist_cols * nbins
    row_len = hist_cols + nbins

    #Allocate vector for integral histogram added row and column
    integral = np.zeros(hist_len*nchannels)
    #Zero out the additional row
    integral[:row_len] = 0
    
    #Zero out the additional column
    for i in range(hist_rows):
        integral[i*row_len: i*row_len+nbins] = 0

    #Split each channel
    channels=[]
    for i in range(nchannels):
        channel = im[:,i]
        channels.append(channel)

    for i in range(nchannels):
        integral_channel = integral[i*hist_len:(i+1)*hist_len-1] #integral on one channel
        wavefrontScan(channels[i],integral_channel,nbins) 
    return integral #1d array




def wavefrontScan(im, integral,nbins): #im: 2d, integral: 1d, IH:
    rows, cols = im.shape
    row_len = (row+1)*nbins
    hy, hx = 0, 0


    for y in range(rows):
        im_row = im[y,:]
        for x in range(cols):
            #Calculate histogram coordinates in a shape of an index
            h00 = hx +           hy
            h01 = hx +           hy + row_len
            h10 = hx + nbins  + hy
            h11 = hx + nbins  + hy + row_len

            #Sum left, upper and upper-left histogram
            sumHist(h00, h01, h10, h11, nbins, integral)
            
            #Calculate bin index of the current point 
            bin_index = im_row[x] * (nbins - 1)

            #Add the current pixel's bin
            integral[h11+bin_index]+=1

            hx += nbins
            hy += row_len



def sumHist(h00, h01, h10, h11,nbins, integral): #hxx : index; #nbins : int, integral : narray 1d
    integral[h11:h11+nbins] = integral[h01:h01+nbins] + integral[h10:h10+nbins] - integral[h00:h00+nbins]

def regionHistogram(integral, region, IH) #integral : 1d
    rows, cols, nchannels, nbins = IH
    hist_rows = rows +1
    hist_cols = cols +1
    hist_len = hist_rows + hist_cols + nbins
    
    x,y,width,height = region



    out = np.zeros(nbins*nchannels)
    for ch in range(nchannels):
        x0 = x
        x1 = x + width
        y0 = y
        y1 = y + height

        row_len = (cols+1)*nbins
        h00 = x0*nbins + y0*row_len
        h01 = x1*mNBins + y0 * row_len
        h10 = x0 * mNBins + y1 * row_len
        h11 = x1 * mNBins + y1 * row_len

        out_ch = out[ch*nbins: (ch+1)*nbins-1]

        regionHist(h00,h01,h10,h11, nbins, nchannels, integral, out_ch)
    
    return out

def regionHist(h00,h01,h10,h11, nbins, nchannels, integral, out_ch):
    out_ch[:] = integral[h11:h11+nbins] - integral[h01:h01+nbins] - integral[h10:h10+nbins] + integral[h00:h00+nbins]

