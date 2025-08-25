import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np


def thresholding():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    histogram = cv.calcHist([imgGray],[0], None,[256],[0,256])

    plt.figure()
    plt.subplot(2,3,1)
    plt.plot(histogram)

    

    thresholdOptions= [cv.THRESH_BINARY,cv.THRESH_BINARY_INV,cv.THRESH_TOZERO,cv.THRESH_TOZERO_INV,cv.THRESH_TRUNC]
    thresholdNames = ['binary', 'binary inverse', 'tozero', 'tozero inverse', 'truncated']

    for i in range(len(thresholdNames)):
        plt.subplot(2,3,i+2)
        _,imgThresh = cv.threshold(imgGray,140,255,thresholdOptions[i])
        plt.imshow(imgThresh,cmap='gray')
        plt.title(thresholdNames[i])


    plt.show()


if __name__ == "__main__":
    thresholding()