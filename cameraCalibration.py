# code from: https://qiita.com/ReoNagai/items/5da95dea149c66ddbbdd
# 内蔵カメラからのキャリブレーション仕様から、画像ファイル読み込みでの判定に改造

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np


square_size = 2.7      # 正方形の1辺のサイズ[cm]
pattern_size = (7, 7)  # 交差ポイントの数

reference_img = 40  # 参照画像の枚数

pattern_points = np.zeros((np.prod(pattern_size), 3),
                          np.float32)  # チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints = []

capture = cv2.VideoCapture("src/mov/calibration_pc.mp4")

totalframecount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    # 総フレーム数
reference_frame = int(totalframecount / reference_img)   # 何フレームごとに判定するか
current_frame = 1   # 現在のフレーム数
fps = int(capture.get(cv2.CAP_PROP_FPS))    # 動画のfps

print("totalframecount: " + str(totalframecount))
print("reference_frame: " + str(reference_frame))

while len(objpoints) < reference_img:
    # 画像の取得
    ret, img = capture.read()
    height = img.shape[0]
    width = img.shape[1]

    if current_frame % reference_frame == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # チェスボードのコーナーを検出
        ret, corner = cv2.findChessboardCorners(gray, pattern_size)
        # コーナーがあれば
        if ret == True:
            print("detected coner!")
            print(str(len(objpoints)+1) + "/" + str(reference_img))
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), term)
            # appendメソッド：リストの最後に因数のオブジェクトを追加
            imgpoints.append(corner.reshape(-1, 2))
            objpoints.append(pattern_points)

    cv2.imshow('image', img)
    # 動画のfps分待機
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break
    current_frame += 1

print("calculating camera parameter...")
# 内部パラメータを計算
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# 計算結果を保存
np.save("mtx", mtx)  # カメラ行列
np.save("dist", dist.ravel())  # 歪みパラメータ
# 計算結果を表示
print("RMS = ", ret)
print("mtx = \n", mtx)
print("dist = ", dist.ravel())
