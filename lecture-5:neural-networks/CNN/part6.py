import cv2 as cv
import os
import matplotlib.pyplot as plt


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





if __name__== "__main__":
    # imageGradient()
    cannyEdge()