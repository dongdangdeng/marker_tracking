import numpy as np
import cv2
from cv2 import aruco
import pandas as pd
import matplotlib.colors as mcolors
import random
import tqdm

random.seed(0)

IS_DEBUG = False

input_video_path = "src/mov/VibrationTest/oricon_g_hd_240.mp4"
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
    if IS_DEBUG and current_frame == 10:
        print("stop")
    ret, img = cap.read()
    if not ret:
        break

    # マーカー認識
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)

    # if current_frame == 1175:
    #     from IPython.core.debugger import Pdb; Pdb().set_trace()

    # 同じidのマーカーが複数検知された際の処理（前回検知されたmarkerとの距離が近い方のみを残す）
    # TODO 一応動いているがデータあっているか検証
    if ids is not None:
        id_list = list(map(lambda id: id[0], ids))   # idsの値をそのまま平のリストに変換
    duplicate_ids = [x for x in set(
        id_list) if id_list.count(x) > 1]   # 重複しているidがあれば抜き出し
    if len(duplicate_ids) > 0:
        for id in duplicate_ids:
            IS_DEBUG and print("duplicate id: " + str(id))
            cpxs = markers_hist[id]['cpx'].dropna().dropna()
            cpys = markers_hist[id]['cpy'].dropna().dropna()
            if len(cpxs) != 0 and len(cpys) != 0:   # 座標にNaNしかない場合（初めてW誤検知した場合等）は処理しない
                last_cpx = list(cpxs[-1:])[0]
                last_cpy = list(cpys[-1:])[0]

                corners_indexes_from_duplicate_id = [i for i, x in enumerate(
                    ids) if x == id]  # 重複idのデータが入っているcornersのindexリスト
                IS_DEBUG and print("duplicate_indexes: ",
                                   corners_indexes_from_duplicate_id)
                # (last_cpx, last_cpy)と今検知された座標の距離のリスト（index=該当のcornersのindex）
                distances = pd.Series(dtype=float)
                for corner_id in corners_indexes_from_duplicate_id:
                    # cpx座標とcpy座標の距離
                    distance_x, distance_y = np.average(
                        corners[corner_id][0], axis=0).astype(int) - [last_cpx, last_cpy]
                    distances.at[corner_id] = np.sqrt(
                        distance_x ** 2 + distance_y ** 2)    # 各xy座標の距離から2点間の距離に変換し、リストに追加
                drop_indexes = distances.index.drop(
                    labels=[distances.idxmin()]).tolist()
                corners = np.delete(corners, drop_indexes, axis=0)
                ids = np.delete(ids, drop_indexes, axis=0)

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
            cpx, cpy = np.average(this_corner, axis=0).astype(int)
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
            markers_current = pd.concat([markers_tmp[0], markers_tmp[0]], axis=1, keys=[
                                        ids[0][0], DELETE_KEY])  # 後で消す方のkey=DELETE_KEY

        else:
            markers_current = pd.concat(
                markers_tmp, axis=1, keys=np.squeeze(ids))

        IS_DEBUG and print(markers_current)
        markers_hist = pd.concat(
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
markers_hist.to_csv(output_path)
print("out put parsed markers '" + str(output_path) + "'")
