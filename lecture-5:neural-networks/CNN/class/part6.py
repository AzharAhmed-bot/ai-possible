import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np


def callback():
    pass


def imageGradient():
    root  = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)


 
    
    laplacian = cv.Laplacian(img,cv.CV_64F,ksize=21)

    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(laplacian,cmap='gray')
    

    # 1 > 1st derivative in x direction
    # 0 > No derivaive in y direction
    # 3 > kernel size -> odd number
    kx,ky = cv.getDerivKernels(1,0,3)
    sobelXKernel = ky @ kx.T
    # print(sobelXKernel)
    sobelX= cv.Sobel(img,cv.CV_64F,1,0,ksize=21)

    plt.subplot(2,3,2)
    plt.imshow(sobelX,cmap='gray')



    sobelY = cv.Sobel(img,cv.CV_64F,0,1,ksize=21)

    plt.subplot(2,3,3)
    plt.imshow(sobelY,cmap='gray')
    plt.show()


def cannyEdge():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    height, width, _ = img.shape
    scale = 1/5

    img = cv.resize(img,(int(width*scale),int(height*scale)),interpolation=cv.INTER_LINEAR)
    winname = 'canny'
    cv.namedWindow(winname)
    cv.createTrackbar('minThresh',winname,0,255,callback)
    cv.createTrackbar('maxThresh',winname,0,255,callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        minThresh = cv.getTrackbarPos('minThresh',winname)
        maxThresh = cv.getTrackbarPos('maxThresh',winname)
        cannyEdge= cv.Canny(img,minThresh,maxThresh)
        cv.imshow(winname,cannyEdge)
    
    cv.destroyAllWindows()



def houghLineTransform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/Jaguar.jpg')
    img = cv.imread(imgPath)
    imgBlur = cv.GaussianBlur(img,(21,21),2) # To reduce noise when getting the edges
    cannyEdge = cv.Canny(imgBlur,50,180)

    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img)

    plt.subplot(2,3,2)
    plt.imshow(imgBlur)

    plt.subplot(2,3,3)
    plt.imshow(cannyEdge, cmap='gray')


    distResol = 1
    angleResol = np.pi /180
    threshold = 150
    lines = cv.HoughLines(cannyEdge, distResol, angleResol, threshold)
    k = 3000

    for curLines in lines:
        rho,theta = curLines[0]
        a =  np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + k*b)
        y1 = int(y0 - k*a)
        x2 = int(x0 - k*b)
        y2 = int(y0 + k*a)
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        
    plt.subplot(2,3,4)
    plt.imshow(img)
    plt.show()




if __name__== "__main__":
    # imageGradient()
    # cannyEdge()
    houghLineTransform()