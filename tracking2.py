import numpy as np
import cv2
from cv2 import aruco
import pandas as pd
import tqdm
import datetime
import sys

IS_DEBUG = False            # デバッグモード
IS_INTERPOLATE = True       # 欠損値の補完を行うか
INTERPOLATE_METHOD = {
    0: "linear", 
    1: "spline"
    }[0]                    # 補完の種類
IS_APPLY_FILTERS = True     # フィルタを適用するか

# 解析する動画のパス
input_video_path = "src/mov/VibrationTest/base_4k_60.MOV"

# 動画のmaxサイズ（これを超えるとリサイズ処理）
h_max = 1080

print("loading '" + input_video_path + "'")
cap = cv2.VideoCapture(input_video_path)
if not (cap.isOpened()):
    print("video loading error")
    sys.exit()

w_origin = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_origin = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 動画のheightがh_maxより大きければh_maxに合わせてリサイズ
if h_origin > h_max:
    h = h_max
    w = int(w_origin * (h / h_origin))

fps = int(cap.get(cv2.CAP_PROP_FPS))    # 動画のfps
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 総フレーム数

# 日付取得（ファイル名用）
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
# YYYYMMDDhhmmss形式に書式化
d = now.strftime('%Y%m%d%H%M%S')

# 出力csvのパス
output_path = "output/test/markers_" + d + ".csv"

# マーカーサイズ
marker_length = 0.044  # [m]

# マーカーの辞書選択
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# キャリブレーションデータの選択
camera_matrix = np.load("calibration/iphonese/mtx.npy")
distortion_coeff = np.load("calibration/iphonese/dist.npy")

# DFの不要な行（後で消す）につけるkey名称
DELETE_KEY = -1

# マーカーの履歴（初期状態でマルチカラムのDFをconcatしてもエラーにならないように最低限フォーマットを整える）
markers_hist = pd.DataFrame([np.nan], columns=[[DELETE_KEY], [0]])

# 鮮鋭化フィルタ
def applySharpFilter(img, k=2):
    sharp_kernel = np.array([
        [-k / 9, -k / 9, -k / 9],
        [-k / 9, 1 + 8 * k / 9, k / 9],
        [-k / 9, -k / 9, -k / 9]
    ], np.float32)
    img_sp = cv2.filter2D(img, -1, sharp_kernel).astype("uint8")
    return img_sp

# バイラテラルフィルタ
def applyBltrlFilter(img, d=15):
    img_bltrl = cv2.bilateralFilter(src=img, d=d, sigmaColor=75, sigmaSpace=75)
    return img_bltrl

# メディアンフィルタ
def applyMedianFilter(img, d=1):
    return cv2.medianBlur(img, d)

"""
cornersとidsの差分を追加したcornersとidsを返す。
"""
def addNewMarkers(current_ids, current_corners, new_corners=None, new_ids=None):
    current_id_list = [] if current_ids is None else list(map(lambda id : id[0] , current_ids))
    current_corners_list = [] if current_ids is None else list(current_corners)
    if new_ids is None: # 追加する方のデータ（new_~)がなにもない場合、currentを次元削減したものだけを返す
        return current_id_list, current_corners_list
    else:
        new_ids_list = list(map(lambda id : id[0] , new_ids))
        origin_ids = [id for id in new_ids_list if id not in current_id_list]   # フィルター適用後のみ検出されたid
        origin_id_indexes = [new_ids_list.index(i) for i in origin_ids]   # origin_idsのidに対応するindex
        added_id_list = np.append(current_id_list, origin_ids)  # id_listに新しく検出されたidを追加
        added_corners_list = current_corners_list
        for i in origin_id_indexes: # cornersに新しく検出された座標を追加
            added_corners_list.append(new_corners[i])
        return added_id_list, added_corners_list

print("parsing markers...")
for current_frame in tqdm.tqdm(range(1, total_frame + 1)):
    IS_DEBUG and print(f"frame: {current_frame}")
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, dsize=(w, h))
    
    # マーカー認識
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)

    # 同じidのマーカーが複数検知された際の処理（前回検知されたmarkerとの距離が近い方のみを残す）
    if ids is not None :
        id_list = list(map(lambda id : int(id[0]) , ids))   # idsの値をそのまま平のリストに変換
        duplicate_ids = [x for x in set(id_list) if id_list.count(x) > 1]   # 重複しているidがあれば抜き出し
    else:
        id_list = []
        duplicate_ids = []
        
    if len(duplicate_ids) > 0:
        for id in duplicate_ids:
            IS_DEBUG and print(f"duplicate id: {id}")
            cpxs = markers_hist[id]['cpx'].dropna()
            cpys = markers_hist[id]['cpy'].dropna()
            if len(cpxs) != 0 and len(cpys) != 0:   # 座標にNaNしかない場合（初めてW誤検知した場合等）は処理しない
                last_cpx = list(cpxs[-1:])[0]
                last_cpy = list(cpys[-1:])[0]
                
                corners_indexes_from_duplicate_id = [i for i, x in enumerate(ids) if x == id] # 重複idのデータが入っているcornersのindexリスト
                IS_DEBUG and print("duplicate_indexes: ", corners_indexes_from_duplicate_id)
                distances = pd.Series(dtype=float)  # (last_cpx, last_cpy)と今検知された座標の距離のリスト（index=該当のcornersのindex）
                for corner_id in corners_indexes_from_duplicate_id:
                    # cpx座標とcpy座標の距離
                    distance_x, distance_y = np.average(corners[corner_id][0], axis=0).astype(int) - [last_cpx, last_cpy]
                    distances.at[corner_id] = np.sqrt(distance_x ** 2 + distance_y ** 2)    # 各xy座標の距離から2点間の距離に変換し、リストに追加
                drop_indexes = distances.index.drop(labels=[distances.idxmin()]).tolist()
                corners = np.delete(corners, drop_indexes, axis=0)
                ids = np.delete(ids, drop_indexes, axis=0)
                id_list = list(map(lambda id : id[0] , ids)) # TODO DRY

    # フィルター適用処理
    # フィルター適用後のimgでのみ検出されたマーカーを追加
    if IS_APPLY_FILTERS:
        #バイラテラルフィルタ
        img_b = applyBltrlFilter(img, 15)
        # 鮮鋭化
        img_s = applySharpFilter(img_b, 2)
        corners_s, ids_s, _ = aruco.detectMarkers(img_s, dictionary)
        id_list, corners_list = addNewMarkers(ids, corners, corners_s, ids_s)
    else:
         id_list, corners_list = addNewMarkers(ids, corners)

    markers_current = pd.DataFrame()
    markers_tmp = []
    if len(corners_list) > 0:
        # マーカーごとに処理
        for i, corner in enumerate(corners_list):
            id = id_list[i]
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
                print(f"marker id: {id}")
                for key, val in this_marker.iteritems():
                    print(key + " : " + str(val[0]))
        
                
        if len(markers_tmp) < 2:    # マーカーが1つのみ検出された場合
            # concatで複数のマーカーDFを結合してマルチカラムを作成しているため、マーカーDFが1つだけの場合
            # 1つのマーカーを複製して無理やりconcatして、複製した方を最後に削除している
            markers_current = pd.concat([markers_tmp[0], markers_tmp[0]], axis=1, keys=[
                                        id_list[0], DELETE_KEY])  # 後で消す方のkey=DELETE_KEY
        else:
            markers_current = pd.concat(
                markers_tmp, axis=1, keys=id_list)

        IS_DEBUG and print(markers_current)
        markers_hist = pd.concat(
            [markers_hist, markers_current], ignore_index=True)
    else:  # マーカーが1つも検出できなかった場合
        markers_hist.loc[current_frame] = np.nan

# 全フレーム終了後の処理

markers_hist.sort_index(level=0, axis=1, inplace=True)

if DELETE_KEY in markers_hist.columns:  # 不要な行を削除
    markers_hist = markers_hist.drop(DELETE_KEY, axis=1)

# 欠損値を補完
if IS_INTERPOLATE:
    print(f"Complementing missing values by {INTERPOLATE_METHOD} method...")
    for col in tqdm.tqdm(markers_hist.columns):
        if (total_frame - markers_hist[col].isnull().sum()) >= 4:
            markers_hist[col] = markers_hist[col].interpolate(
                method=INTERPOLATE_METHOD, order=2, limit_direction="both")
            if (col[1] == "cpx" or col[1] == "cpy") and ~np.isnan(markers_hist[col][0]):     # cpx、cpyはintへ変換
                markers_hist[col] = markers_hist[col].astype(int)

# カラムのidをintに変換
markers_hist.columns = markers_hist.columns.set_levels(
    markers_hist.columns.levels[0].astype(int),
    level=0
    )

markers_hist.to_csv(output_path)
print(f"out put parsed markers '{output_path}'")
