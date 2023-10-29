from integral_histogram import *
import cv2 as cv

N_BINS = 20
N_CHANNELS=3

def compare_X2(hist1,hist2,lenght):
    sin = 0
    for i in range(lenght):
        val = hist1[i] - hist2[i]
        if val!=0:
            sin += (val*val)/(hist1[i]+hist2[i])
    return sin



A = cv.imread("data/ima.tiff")
B = cv.imread("data/imb.tiff")

A_height,A_width, A_channels = np.shape(A)
B_height, B_width, B_channels = np.shape(B)
 
#!!! fonction catch if channel!=N_CHANNEl
    
#Calculate integral histogram for each image
integralA = integralHistogram(A_height, A_width, N_CHANNELS,N_BINS, A)
integralB = integralHistogram(B_height, B_width, N_CHANNELS,N_BINS, B)

#Now we can calculate histogram for any window position and size very fast in O(N_BINS)
region = (0, 0, 100, 100) #x,y,width,height

out = regionHistogram(integralA, region, A_height, A_width, N_CHANNELS, N_BINS) #shape = (nbins*nchannels, 1)

#Print out histogram values
#For color images the histograms for each color channel are stacked 
#one after the other. This way we get N_BINS * N_CHANNELS values.
'''
for i in range(len(out)):
    print(out[i])
'''
print(out)

#Compare the images using histograms in sliding window.
#The output image type is float
#We've got some interesting results with LTP of the actual images
#We implemented the X^2 distance measure (function compHist()).

sim = compare(integralA,integralB , [20,20], compare_X2, out)

cv.normalize(sim, sim, 0, 1, cv.NORM_MINMAX)
cv.imshow("Similarity", sim)
