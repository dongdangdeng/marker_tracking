import cv2
from PIL import Image
import numpy as np

file = "./src/img/trackingtest01.jpg"

img = cv2.imread(file)

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
outputImg = "./output/test/result_corner"+".png"


def drawCorner(argImg):
    argImg = cv2.cvtColor(argImg, cv2.COLOR_BGR2RGB)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(argImg, dictionary)
    print("corners:" + str(corners))
    print("ids:" + str(ids))
    print("rejectedImgPoints:" + str(rejectedImgPoints))

    # IDを書く
    aruco.drawDetectedMarkers(argImg, corners, ids, (0, 255, 0))

    for i, corner in enumerate(corners):
        points = corner[0].astype(np.int32)
        # マーカーの輪郭の検出
        cv2.polylines(argImg, [points], True, (255, 0, 0))
        cv2.putText(argImg, str(points[0]), tuple(
            points[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(argImg, str(points[1]), tuple(
            points[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(argImg, str(points[2]), tuple(
            points[2]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(argImg, str(points[3]), tuple(
            points[3]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    return argImg, corners


img, corners = drawCorner(img)

print(corners[0])

cv2.startWindowThread()
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 640, 480)
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
