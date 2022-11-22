import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

for i in range(50):
    ar_image = aruco.drawMarker(dictionary, i, 150)
    fileName = "marker/ar" + str(i).zfill(2) + ".png"
    cv2.imwrite(fileName, ar_image)
