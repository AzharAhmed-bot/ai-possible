import cv2 as cv
import numpy as np


height,width = 480,640

cap = cv.VideoCapture('/dev/video0', cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

def preProcessing(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 3)
    canny = cv.Canny(imgBlur, 50, 100)
    kernel = np.ones((5, 5))
    imgDial = cv.dilate(canny, kernel, iterations=2)
    imgThres = cv.erode(imgDial, kernel, iterations=1)
    return imgThres

def getContours(img, drawImg):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxArea = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 2000:
            cv.drawContours(drawImg, cnt, -1, (255, 0, 0), 2)  
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv.drawContours(drawImg,biggest,-1,(255,0,0),20)
    return biggest

def reOrder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew



def getWrap(img, biggest):
    biggest = reOrder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(img, matrix, (width, height))

    imgCropped =imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv.resize(imgCropped, (width, height))
    return imgCropped


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (width, height), interpolation=cv.INTER_LANCZOS4)

    imgContours = frame.copy()
    result = preProcessing(frame)
    biggest = getContours(result, imgContours)
    if biggest.size != 0:
        imgWraped = getWrap(frame, biggest)

    # cv.imshow('Result', result)
    cv.imshow('Contours', imgContours) 
    cv.imshow('Wrap', imgWraped)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
