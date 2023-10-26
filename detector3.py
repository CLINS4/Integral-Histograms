from integral_histogram import *
import cv2 as cv

N_BINS = 3
N_CHANNELS=20

def compare_X2(hist1,hist2,lenght):
    sin = 0
    for i in range(lenght):
        val = hist1[i] - hist2[i]
        if val!=0:
            sin += (val*val)/(hist1[i]+hist2[i])
    return sin


def main:
    A = imread("ima.tiff")
    B = imread("imb.tiff")
    IHa = construct_IntegralHistogram(A.shape(),N_CHANNELS,N_BINS))
    IHb = construct_IntegralHistogram(B.shape(),N_CHANNELS,N_BINS))
    
    
    #Calculate integral histogram for each image
    integralA = integralHistogram(IHa)
    integralB = integralHistogram(IHb)

    #Now we can calculate histogram for any window position and size very fast in O(N_BINS)
    region = (0, 0, 100, 100) #x,y,width,height

    out = regionHistogram(histA, region)

