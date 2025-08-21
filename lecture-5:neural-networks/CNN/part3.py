import cv2 as cv
import os 
import matplotlib.pyplot as plt


def imageResize():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img =cv.imread(imgPath)
    # y by x
    img= img[190:330,170:440,:]
    height, width, _ = img.shape

    scale =4

    interpMethods=[
        cv.INTER_AREA,
        cv.INTER_LINEAR,
        cv.INTER_NEAREST,
        cv.INTER_CUBIC,
        cv.INTER_LANCZOS4
    ]

    interpTitles=[
        "INTER_AREA",
        "INTER_LINEAR",
        "INTER_NEAREST",
        "INTER_CUBIC",
        "INTER_LANCZOS4"
    ]

    
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title("Original")

    for i in range(len(interpMethods)):
        imgResize = cv.resize(img, (int(width*scale), int(height*scale)), interpolation=interpMethods[i])
        plt.subplot(2,3,i+2)
        plt.imshow(imgResize)
        plt.title(interpTitles[i])

    plt.tight_layout()
    plt.show()




def grayhistogram():
    root = os.getcwd()
    imgPath =  os.path.join(root, 'Images/cat.jpg')
    img =  cv.imread(imgPath,cv.IMREAD_GRAYSCALE)

    plt.figure()
    plt.imshow(img,cmap='gray')

    histogram = cv.calcHist([img],[0],None,[256],[0,256])

    plt.figure()
    plt.plot(histogram)
    plt.xlabel("bins")
    plt.ylabel("# of pixels")
    plt.show()


def colorHistogram():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')

    img = cv.imread(imgPath)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img_RGB)
    plt.show()

    colorList = ['b', 'g', 'r']

    for i in range(len(colorList)):
        histrogram = cv.calcHist([img_RGB],[i],None,[256],[0,256])

        plt.plot(histrogram, colorList[i])

    plt.xlabel("pixel intensity")
    plt.ylabel("# of pixels")
    plt.show()

def histogramRegion():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img= cv.imread(imgPath)
    img_RGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img_RGB =img_RGB[190:330,170:440,:]


    plt.figure()
    plt.imshow(img_RGB)
    plt.show()

    colorList = ['b','g','r']

    for i in range(len(colorList)):
        histogram = cv.calcHist([img_RGB],[i],None,[256],[0,256])

        plt.plot(histogram,colorList[i])

    plt.xlabel("pixel intensity")
    plt.ylabel("# of pixels")
    plt.show()


if __name__ == '__main__':
    imageResize()
    # grayhistogram()
    # colorHistogram()
    # histogramRegion()