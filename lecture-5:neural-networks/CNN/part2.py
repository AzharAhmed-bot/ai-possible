import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def pureColors():
    zeros = np.zeros((100,100))
    ones = np.ones((100,100))
    blueImage = cv.merge((zeros, zeros,ones*255))
    redImage = cv.merge((ones*255, zeros, zeros))
    greenImage = cv.merge((zeros, ones*255, zeros))
    print(blueImage)
    
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(blueImage)

    plt.subplot(2,3,2)
    plt.imshow(redImage)

    plt.subplot(2,3,3)
    plt.imshow(greenImage)
    plt.show()



def bgrChannelGrayScale():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)
    r,g,b = cv.split(img)


    plt.figure()

    plt.subplot(2,3,1)
    plt.imshow(b,cmap='gray')


    plt.subplot(2,3,2)
    plt.imshow(g)

    plt.subplot(2,3,3)
    plt.imshow(r)
    
    plt.show()

def bgrChannelColor():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img =cv.imread(imgPath)

    b,g,r = cv.split(img)
    zeros = np.zeros_like(b)
    blueImage = cv.merge((b,zeros,zeros))
    greenImage = cv.merge((zeros,g, zeros))
    redImage = cv.merge((zeros, zeros, r))

    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(blueImage)

    plt.subplot(2,3,2)
    plt.imshow(greenImage)

    plt.subplot(2,3,3)
    plt.imshow(redImage)
    plt.show()



# Gray scale is used to reduce the amount of data
# Grayscale pixel = 0.299R + 0.587G + 0.114B
def grayscale():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    # img = cv.imread(imgPath)
    # imgGrayscale= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('img',imgGrayscale)
    # cv.waitKey(0)
    # ---------------or ------------------
    img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
    cv.imshow('img',img)
    cv.waitKey(0)


# HSV 
# Hue = Type of color
# Saturation = Concentration of the color
# Value = Brightness
def hsvColorSegmentation():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lowerbound = np.array([0,0,40])
    upperbound = np.array([18,62,79])

    mask = cv.inRange(hsv, lowerbound, upperbound)

    # plt.figure()
    # plt.imshow(imgRGB)
    # plt.show()

    cv.imshow('mask', mask)
    cv.waitKey(0)



if __name__ == "__main__":
    # pureColors()
    # bgrChannelGrayScale()
    bgrChannelColor()
    # grayscale()
    # hsvColorSegmentation()