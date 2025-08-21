import cv2 as cv
import os
import matplotlib.pyplot as plt
img = None  # define globally

def readImage():
    global img
    root = os.getcwd()
    imgPath = os.path.join(root, 'lecture-5:neural-networks/CNN/Images/cat.jpg')
    img = cv.imread(imgPath)
    debug = 1
    print(img)
    cv.imshow('img', img)
    cv.waitKey(0)



def writeImage():
    root = os.getcwd()
    imgPath = os.path.join(root, 'lecture-5:neural-networks/CNN/Images/cat.jpg')
    img = cv.imread(imgPath)
    outPath = os.path.join(root, 'lecture-5:neural-networks/CNN/Images/cat_out.jpg')
    cv.imwrite(outPath, img)


def videoFromWebcam():
    cap = cv.VideoCapture('/dev/video0', cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv.imshow("Webcam", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def videoFromFile():
    root = os.getcwd()
    vidPath = os.path.join(root, 'lecture-5:neural-networks/CNN/Videos/cat.mp4')
    cap = cv.VideoCapture(vidPath)

    while cap.isOpened():

        ret, frame =cap.read()
        if ret:
            cv.imshow('video',frame)
            delay =int(1000/60)
            if cv.waitKey(delay) == ord('q'):
                break


def writeVideoToFile():
    cap = cv.VideoCapture('/dev/video0', cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)


    root = os.getcwd()
    output = os.path.join(root, 'lecture-5:neural-networks/CNN/Videos/webcam.avi')

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(output,fourcc, 20.0, (640,480))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            out.write(frame)
            cv.imshow('webcam',frame)

        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    out.release()
    cv.destroyAllWindows()



def readWritePixelValues():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

    imgRGB[290,350] =(255,0,0)
    # print(eyePixel)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()



def readAndWritePixelRegion():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/cat.jpg')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

    eyeRegion = imgRGB[200:244,241:280]
    print(eyeRegion)

    dx = 244 - 200
    dy = 280 - 241
    
    imgRGB[100:100+dx, 180:180+dy] = eyeRegion

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()


    



if __name__ == "__main__":
    # readImage()
    # writeImage()
    # videoFromWebcam()
    # videoFromFile()
    # writeVideoToFile()
    # readWritePixelValues()
    readAndWritePixelRegion()