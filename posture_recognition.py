# code from: https://qiita.com/ReoNagai/items/a8fdee89b1686ec31d10

#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import cv2
from cv2 import aruco
import pandas as pd


def main():
    cap = cv2.VideoCapture("src/mov/test01.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))    # 動画のfps
    current_frame = 1   # 現在の動画フレーム
    # マーカーサイズ
    marker_length = 0.031  # [m]
    # マーカーの辞書選択
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # キャリブレーションデータの選択
    camera_matrix = np.load("calibration/iphonese/mtx.npy")
    distortion_coeff = np.load("calibration/iphonese/dist.npy")

    # マーカーの履歴
    markers_hist = pd.DataFrame()

    while True:
        ret, img = cap.read()
        if not ret or cv2.waitKey(int(1000 / fps)) == 27:
            break

        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
        # 可視化
        aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 255))

        markers = pd.DataFrame()
        if len(corners) > 0:
            # マーカーごとに処理
            for i, corner in enumerate(corners):
                id = ids[i][0]
                this_corner = np.squeeze(corners)[id]

                # 姿勢認識
                # rvec -> rotation vector, tvec -> translation vector
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corner, marker_length, camera_matrix, distortion_coeff)

                # < rodoriguesからeuluerへの変換 >

                # 不要なaxisを除去
                tvec = np.squeeze(tvec)
                rvec = np.squeeze(rvec)
                # 回転ベクトルからrodoriguesへ変換
                rvec_matrix = cv2.Rodrigues(rvec)
                rvec_matrix = rvec_matrix[0]  # rodoriguesから抜き出し
                # 並進ベクトルの転置
                transpose_tvec = tvec[np.newaxis, :].T
                # 合成
                proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                # オイラー角への変換
                euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[
                    6]  # [deg]

                # 中心座標(ピクセル単位)
                cpx = int((this_corner[0][0] + this_corner[2][0]) / 2)
                cpy = int((this_corner[0][1] + this_corner[2][1]) / 2)

                # マーカー座標をmarkers(DataFrame)に書き込み
                marker_info = {"x": tvec[0], "y": tvec[1], "z": tvec[2],
                               "roll": euler_angle[0][0], "pitch": euler_angle[1][0], "yaw": euler_angle[2][0],
                               "cpx": cpx, "cpy": cpy}
                markers[id] = [marker_info.items()]

                if True:
                    print("marker id: " + str(id))
                    for key, val in marker_info.items():
                        print(key + " : " + str(val))

                # 可視化
                draw_pole_length = marker_length/2  # 現実での長さ[m]
                cv2.drawFrameAxes(img, camera_matrix, distortion_coeff,
                                  rvec, tvec, draw_pole_length)
        print(markers)
        break
        cv2.imshow('drawDetectedMarkers', img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
