import cv2 as cv
import numpy as np


def integralHistogram(height, width, nchannels, nbins,im):
    rows,cols = height, width 
    hist_rows = rows +1 #add a row to avoid to write so much initial condition in order to create the 1st row in propagation step
    hist_cols = cols +1 #add a col "    "   "   "       "   "   "   "   "   "   "   "   "   "   "   "   " col
    hist_len = hist_rows * hist_cols * nbins
    #row_len = hist_cols + nbins

    #Allocate vector for integral histogram added row and column
    integral = np.zeros(hist_len*nchannels).reshape(nchannels, hist_rows, hist_cols, nbins)

    #Split each channel
    for i in range(nchannels):
        channel = im[:,:,i]
        print("channel ",i)
        wavefrontScan2(channel,integral[i],nbins) 
    return integral #4d array 




def wavefrontScan(im, integral,nbins): #im: 2d (rows, cols), integral: 1d
    rows, cols = im.shape
    row_len = (rows+1)*nbins
    hy, hx = 0, 0
    
    integral.reshape(rows+1, cols+1, nbins)

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
            if im_row[x] == 0:
                bin_index = 0
            else:
                bin_index = int(np.ceil(im_row[x] * nbins / 255) -1)

            #print(integral[h11+bin_index])
            #Add the current pixel's bin
            integral[h11+bin_index]+=1
            
            #print(bin_index, integral[h11+bin_index], im_row[x])

            hx += nbins
            hy += row_len

def wavefrontScan2(im, integral,nbins): #im: 2d (rows, cols), integral: 3d (hist_rows, hist_cols, bins)
    rows, cols = im.shape
    #row_len = (rows+1)*nbins
    #hy, hx = 0, 0
    
    #integral.reshape((rows+1, cols+1,nbins))

    for x in range(rows-1): #dimensions reversed from the paper because of the dimension management of python array (in paper: col=1st dim and row=2nd dim whereas in python: row = 1st dim and col = 2nd dim)
        for y in range(cols-1):
            
            #Calculate histogram coordinates in a shape of an index
            h00 = integral[x,y,:]
            h01 = integral[x,y+1,:]
            h10 = integral[x+1,y,:]
            h11 = integral[x+1,y+1,:]

            

            #Sum left, upper and upper-left histogram
            sumHist2(h00, h01, h10, h11, nbins)
            
            #Calculate bin index of the current point 
            bin_index = int(np.ceil(im[x,y] * nbins / 255) -1)

            #print(integral[h11+bin_index])
            #Add the current pixel's bin
            integral[x+1,y+1,bin_index]+=1
            
            


def sumHist(h00, h01, h10, h11,nbins, integral): #hxx : index; #nbins : int, integral : narray 1d
    #print("h11", np.shape(integral[h11:h11+nbins]), h11, h11+nbins)
    #print("h01",np.shape(integral[h01:h01+nbins]), h01, h01+nbins)
    #print("h10",np.shape(integral[h10:h10+nbins]), h10, h10+nbins)
    #print("h00",np.shape(integral[h00:h00+nbins]), h00, h00+nbins)
    if np.shape(integral[h01:h01+nbins]) == 20 and np.shape(integral[h10:h10+nbins]) == 20 and np.shape(integral[h00:h00+nbins]) == 20:
        integral[h11:h11+nbins] = integral[h01:h01+nbins] + integral[h10:h10+nbins] - integral[h00:h00+nbins]

def sumHist2(h00,h01,h10,h11,nbins): # one point of integral 2d, one channel, all bins
    h11 = h01 + h10 - h00

def regionHistogram(integral, region, height, width, nchannels, nbins):
    '''
    integral : 1d
    region : 2d
    height, width, nchannels, nbins : int
    return : 1d, integral for one point
    '''
    rows, cols = height, width
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
        h01 = x1* nbins + y0 * row_len
        h10 = x0 * nbins + y1 * row_len
        h11 = x1 * nbins + y1 * row_len

        out_ch = out[ch*nbins: (ch+1)*nbins-1]

        regionHist(h00,h01,h10,h11, nbins, nchannels, integral, out_ch)
    
    return out

def regionHist(h00,h01,h10,h11, nbins, nchannels, integral, out_ch):
    out_ch[:] = integral[h11:h11+nbins] - integral[h01:h01+nbins] - integral[h10:h10+nbins] + integral[h00:h00+nbins]


def compare(h1, h2, size, cmp, nbins, nchannels, out, ImHeight, ImWidth):
    '''
    h1, h2 : 1d
    size : 2d
    cmp : function
    nbins, nchannels : int
    out
    '''
    Swidth, Sheight = size
    
    out_cols = ImWidth  - Swidth  + 1
    out_rows = ImHeight - Sheight + 1
    hist_rows = ImHeight + 1
    hist_cols = ImWidth + 1
    hist_len = hist_rows * hist_cols * nbins


    channels = []

    #Compare each channel individually
    for i in range(nchannels):
        #Compare single channel
        h1_ch = h1[i*hist_len:(i+1)*hist_len-1]
        h2_ch = h2[i*hist_len:(i+1)*hist_len-1]
        channels.append(compSingle(h1_ch, h2_ch, size, nchannels, nbins, cmp, ImHeight, ImWidth))

    #Merge all channels to single matrix
    out = cv.merge(channels)

    return out
        


def compSingle(h1, h2, size, nchannels ,nbins, cmp, ImHeight,ImWidth):
    #Compares single channel integral histograms
    Swidth,Sheight = size   

    out_cols = ImWidth  - Swidth  + 1
    out_rows = ImHeight - Sheight + 1

    #Pre-calculated values for performance boost
    row_len = (ImWidth + 1) * nbins;
    width = Swidth * nbins;
    height = Sheight * row_len;
    
    hy, hx = 0, 0
    out = np.zeros((out_rows, out_cols))
    for y in range(out_rows):
        p_row = out[y,:]
        for x in range(out_cols):
            #Allocate memory for output histograms
            res1 = np.zeros(nbins)
            res2 = np.zeros(nbins)
            h00 = hx         + hy
            h01 = hx + width + hy
            h10 = hx         + hy + height
            h11 = hx + width + hy + height
            
            regionHist( h00,h01,h10,h11, nbins, nchannels, h1 , res1);
            regionHist( h00,h01,h10,h11, nbins, nchannels, h2 , res2);
            p_row[x] = cmp(res1, res2, nbins); #results added to channel[i] point by point

    return out #return results of comparasion of h1 and h2 with 2d matrix


