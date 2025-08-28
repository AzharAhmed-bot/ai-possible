import cv2 as cv
import numpy as np
import os


# Have a video capture
# Show the mask of only orange color

myColors =[[0,125,186,179,255,255]]

myColorValues = [[0,0,255]]

myPoints = []

def callback():
    pass

def getContours(img,frame):

    contours,hierachy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)
    
        if area > 500:
            area = cv.drawContours(img,cnt,-1,myColors[0][0:3],3)
            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv.boundingRect(approx)
            # cv.rectangle(frame, (x,y),(x+w,y+h), myColors[0][0:3], 3)
            cv.circle(frame,(x,y),10,(0,255,0),cv.FILLED)
            if x!=0 and y!=0:
                myPoints.append([x,y])
    
    return myPoints

def drawOnCanvas(frame,myPoints):
    for point in myPoints:
        cv.circle(frame,(point[0],point[1]),10,(0,255,0),cv.FILLED)
    

def findColors():
    cap = cv.VideoCapture('/dev/video0', cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        imgHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)



        for color in myColors:
            lowerBound = np.array(color[0:3])
            upperBound = np.array(color[3:6])

        mask = cv.inRange(imgHSV, lowerBound, upperBound)

        myPoints = getContours(mask,frame)
        drawOnCanvas(frame,myPoints)
        
        # frame = cv.flip(frame,1)
        cv.imshow('frame', frame)
        # cv.imshow('mask', mask)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


        



if __name__ == '__main__':
    findColors()







