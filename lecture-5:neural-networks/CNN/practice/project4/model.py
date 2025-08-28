import cv2 as cv

width, height = 640, 480
nPlate = cv.CascadeClassifier('Resources/haarcascade_russian_plate_number.xml') 
minArea = 200  

cap = cv.VideoCapture('/dev/video0', cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frameResize = cv.resize(frame, (width, height))
    imgGray = cv.cvtColor(frameResize, cv.COLOR_BGR2GRAY)

    numberPlates = nPlate.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv.putText(frameResize, 'Number Plate', (x, y - 5),
                       cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            cv.rectangle(frameResize, (x, y), (x + w, y + h), (0, 255, 0), 2)
            imgRoi = frameResize[y:y + h, x:x + w]
            cv.imshow('ROI', imgRoi)

    cv.imshow('frame', frameResize)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
