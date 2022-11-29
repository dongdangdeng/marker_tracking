import numpy as np
import cv2
from cv2 import aruco
import pandas as pd
import matplotlib.colors as mcolors
import random
import tqdm

random.seed(0)

IS_DEBUG = True

input_video_path = "src/mov/test06.mp4"
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))    # 動画のfps
# current_frame = 1   # 現在の動画フレーム
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 総フレーム数

# マーカーサイズ
marker_length = 0.0255  # [m]
# マーカーの辞書選択
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# キャリブレーションデータの選択
camera_matrix = np.load("calibration/iphonese/mtx.npy")
distortion_coeff = np.load("calibration/iphonese/dist.npy")

# DFの不要な行（後で消す）につけるkey名称
DELETE_KEY = -1

# 軌跡の長さ
orbit_length = 100

# マーカーの履歴（初期状態でマルチカラムのDFをconcatしてもエラーにならないように最低限フォーマットを整える）
markers_hist = pd.DataFrame([np.nan], columns=[[DELETE_KEY], [0]])

# matplotlibのカラーテーブルを持ってくる（148色分）
colors = list(map(lambda color: tuple(map(lambda c: int(c * 255),
              mcolors.to_rgb(color))), mcolors.CSS4_COLORS.values()))
random.shuffle(colors)

# 出力csvのパス
output_path = "output/test/markers.csv"


print("parsing markers...")
for current_frame in tqdm.tqdm(range(1, total_frame + 1)):
    IS_DEBUG and print("frame: " + str(current_frame))
    if current_frame == 100:
        print("stop")
    ret, img = cap.read()
    if not ret:
        break

    # マーカー認識
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)

    markers_current = pd.DataFrame()
    markers_tmp = []
    if len(corners) > 0:
        # マーカーごとに処理
        for i, corner in enumerate(corners):
            id = ids[i][0]
            this_corner = np.squeeze(corner)

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
            this_marker = pd.DataFrame([{"x": tvec[0], "y": tvec[1], "z": tvec[2],
                                         "roll": euler_angle[0][0], "pitch": euler_angle[1][0], "yaw": euler_angle[2][0],
                                         "cpx": cpx, "cpy": cpy
                                         }])
            markers_tmp.append(this_marker)

            if IS_DEBUG:
                print("marker id: " + str(id))
                for key, val in this_marker.iteritems():
                    print(key + " : " + str(val[0]))

        if len(markers_tmp) < 2:    # マーカーが1つのみ検出された場合
            # concatで複数のマーカーDFを結合してマルチカラムを作成しているため、マーカーDFが1つだけの場合
            # 1つのマーカーを複製して無理やりconcatして、複製した方を最後に削除している
            markers_tmp = [markers_tmp[0], markers_tmp[0]]
            markers_current = pd.concat(markers_tmp, axis=1, keys=[
                                        ids[0][0], DELETE_KEY])  # 後で消す方のkey=DELETE_KEY
        else:
            markers_current = pd.concat(
                markers_tmp, axis=1, keys=np.squeeze(ids))

        markers_hist = pd.concat(   # TODO markers_currentの値がNaNしかない場合、markers_tmpの内容が何であれNaNになってしまう？
            [markers_hist, markers_current], ignore_index=True)
    else:  # マーカーが1つも検出できなかった場合
        markers_hist.loc[current_frame] = np.nan

# 全フレーム終了後の処理

markers_hist.sort_index(level=0, axis=1, inplace=True)

if DELETE_KEY in markers_hist.columns:  # 不要な行を削除
    markers_hist = markers_hist.drop(DELETE_KEY, axis=1)

# 欠損値をスプライン補完
print("Complementing missing values...")
for col in tqdm.tqdm(markers_hist.columns):
    if (total_frame - markers_hist[col].isnull().sum()) >= 4:
        markers_hist[col] = markers_hist[col].interpolate(
            method="spline", order=2, limit_direction="both")
        if (col[1] == "cpx" or col[1] == "cpy") and ~np.isnan(markers_hist[col][0]):
            markers_hist[col] = markers_hist[col].astype(int)

markers_hist
# markers_hist.to_csv(output_path)
# print("out put parsed markers '" + str(output_path) + "'")
