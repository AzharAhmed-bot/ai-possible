import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

def SIFT():
    root = os.getcwd()
    imgPath = os.path.join(root, 'Images/Jaguar.jpg')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    if imgGray is None:
        raise FileNotFoundError(f"Image not found at {imgPath}")

    sift = cv.SIFT_create()
    keypoints = sift.detect(imgGray, None)
    imgKeypoints = cv.drawKeypoints(imgGray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgKeypoints, cmap='gray')
    plt.axis('off')
    plt.show()

def calibrate(showPic=True):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'Images/calibration')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    if not imgPathList:
        raise FileNotFoundError(f"No calibration images found in {calibrationDir}")

    nRows = 8  # number of inner corners per a chessboard row
    nCols = 8  # number of inner corners per a chessboard column
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    for CurImagePath in imgPathList:
        imgBGR = cv.imread(CurImagePath)
        if imgBGR is None:
            print(f"Warning: Could not read image {CurImagePath}, skipping.")
            continue
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)

        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nCols, nRows), None)

        if cornersFound:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)

            if showPic:
                cv.drawChessboardCorners(imgBGR, (nCols, nRows), cornersRefined, cornersFound)
                cv.imshow('ChessBoard', imgBGR)
                cv.waitKey(500)
        else:
            print(f"Chessboard not found in {CurImagePath}")

    cv.destroyAllWindows()

    if not worldPtsList or not imgPtsList:
        raise RuntimeError("No chessboard corners were found in any calibration images.")

    # Calibrate
    ret, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None
    )
    print('Camera Matrix:\n', camMatrix)
    print('Reproj Error (pixels):\n {:.4f}'.format(ret))

    # Save calibration Params
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibrationParams.npz')
    np.savez(paramPath, repError=ret, camMatrix=camMatrix, distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)

    return camMatrix, distCoeff

def runCalibration():
    calibrate(showPic=True)

if __name__ == '__main__':
    # SIFT()
    runCalibration()