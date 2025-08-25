import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt



def convolution2d():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)


    n=100
    kernel = np.ones((n,n),np.float32)/(n*n)
    imgFilter = cv.filter2D(imgRGB,-1,kernel)


    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(imgRGB)

    plt.subplot(2,3,2)
    plt.imshow(imgFilter)
    plt.show()



def callback():
    pass

def averageFiltering():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    winName = 'avg Filtering'
    cv.namedWindow(winName)
    cv.createTrackbar('n',winName,3,100,callback)

    height,width,_= img.shape
    scale = 1/4
    width= int(width*scale)
    height = int(height*scale)

    img= cv.resize(img,(width,height))

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        n = cv.getTrackbarPos('n',winName)
        imgFilter = cv.blur(img,(n,n)) 
        cv.imshow(winName,imgFilter)
    
    cv.destroyAllWindows()  


def medianfiltering():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    noisyImg= imgRGB.copy()

    noiseProb = 0.05
    noise = np.random.rand(noisyImg.shape[0],noisyImg.shape[1])
    noisyImg[noise < noiseProb/2] = 0
    noisyImg[noise > 1-noiseProb/2] = 255


    imgFilter = cv.medianBlur(noisyImg,5)

    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(noisyImg)

    plt.subplot(2,3,2)
    plt.imshow(imgFilter)
    plt.show()



def gaussianKernel(size,sigma):
    kernel = cv.getGaussianKernel(size,sigma)
    kernel = np.outer(kernel,kernel)
    return kernel


def gaussianFiltering():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)

    n = 51
    kernel = gaussianKernel(n,8)

    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(kernel)
    
    ax = fig.add_subplot(2,3,2,projection= '3d')
    x = np. arange(0,n)
    y = np.arange(0,n)
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,kernel,cmap='viridis')
    # plt.show()


    winName ='gaus Filter'
    cv.namedWindow(winName)
    cv.createTrackbar('sigma',winName,1,20,callback)
    height, width, _= img.shape
    scale = 1/4

    width = int(width*scale)
    height = int(height*scale)
    img_resize = cv.resize(img,(width,height))

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        sigma = cv.getTrackbarPos('sigma',winName)
        imgFilter = cv.GaussianBlur(img_resize,(n,n),sigma)
        cv.imshow(winName,imgFilter)
    
    cv.destroyAllWindows()















if __name__ =="__main__":
    # convolution2d()
    # averageFiltering()
    # medianfiltering()
    gaussianFiltering()
    