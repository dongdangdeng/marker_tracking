import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.colors as mcolors
import sys
# import matplotlib.pyplot as plt

random.seed(0)
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


def getCornersFromID(_corners, _ids, _id):
    if not np.any(_ids == [_id]):  # 指定されたidがidsに存在しなければFalseを返す
        return np.nan
    # whereは[[1次元目のindex], [2次元目のindex]] で返ってくるが、ほしいのは1次元目のindexだけなので[0][0]で指定
    index = np.where(_ids == _id)[0][0]
    # _corners[index][0][0~3]にそれぞれ座標が入っているのでタプルにして返す
    return tuple(map(lambda c: tuple(c), _corners[index][0]))


def getCenterPoint(corners):
    # cornersがFalseの場合はFalseを返す（TODO:指定の座標いがはすべてFalseを返すように改良）
    # Trueであれば、左上座標と右下座標を平均した値を返す
    if type(corners) != tuple or np.isnan(corners[0][0]) or np.isnan(corners[2][0]) or np.isnan(corners[0][1]) or np.isnan(corners[2][1]):
        return np.nan
    x = int((corners[0][0] + corners[2][0]) / 2)
    y = int((corners[0][1] + corners[2][1]) / 2)
    return (x, y)


def apllyFilters(img):
    k = 3
    sharp_kernel = np.array([
        [-k / 9, -k / 9, -k / 9],
        [-k / 9, 1 + 8 * k / 9, k / 9],
        [-k / 9, -k / 9, -k / 9]
    ], np.float32)
    img_sp = cv2.filter2D(img, -1, sharp_kernel).astype("uint8")
    return img_sp


cap = cv2.VideoCapture("src/mov/test02.mp4")

if not (cap.isOpened()):   # 正常に読み込めなかった場合終了する（VideoCaptureコンストラクタ自体は失敗してもFalseを返さないので注意）
    sys.exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frameCount = 1
ret, frame = cap.read()
h, w = frame.shape[:2]
cv2.startWindowThread()
corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)
# matplotlibのカラーテーブルを持ってくる（148色分）
colors = list(map(lambda color: tuple(map(lambda c: int(c * 255),
              mcolors.to_rgb(color))), mcolors.CSS4_COLORS.values()))
random.shuffle(colors)
orbit_length = 100

while True:
    # ディレイ＆escキー、フレーム終端チェック
    if not ret or cv2.waitKey(fps) == 27:
        break
    # frame = apllyFilters(frame)

    # マーカー検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)

    # 検出id数が1になるとmapのイテレータが回せなくなるため、最後にNaNを追加（あまり良くない）
    id_list = np.append(np.squeeze(ids).copy(), np.nan)

    # 最後の[:-1]でid_listに追加したnanの探索結果を除外
    centers = list(map(lambda id: getCenterPoint(
        getCornersFromID(corners, ids, id)), id_list))[:-1]
    id_list = id_list[:-1]  # 同様にid_listも最後にNaNが入っているので除外

    if id_list[0] == None:  # マーカーが1つも検知できなかった場合
        orbit = pd.DataFrame([np.nan], index=[frameCount])  # すべてのcolがnanのDF作成
    else:
        # indexがフレーム数、col名が各id、値がセンター位置のDF作成
        orbit = pd.DataFrame([centers], columns=id_list, index=[frameCount])

    if frameCount == 1:  # 最初のフレームの場合、orbitで初期化
        orbits = orbit
    else:               # 次回以降はorbitsにorbitを連結
        orbits = pd.concat([orbits, orbit])

    # 画面情報書き込み
    aruco.drawDetectedMarkers(frame, corners, ids, (255, 255, 0))
    for id, positions in orbits.iteritems():
        id = int(id)

        # 欠損値を補完
        for i in np.arange(2):
            comps = pd.Series(
                list(map(lambda pos: pos[i] if type(pos) == tuple else pos, positions)))
            comps = comps.interpolate(
                "ffill").interpolate("bfill").astype('int')
            if i == 0:
                xs = comps
            else:
                ys = comps

        # 補完したx, y座標を元の(x, y)座標に結合
        positions = np.array(list(zip(xs, ys))[-orbit_length:])

        cv2.polylines(frame, [positions], False, colors[id], 1)

    # フレーム描画
    cv2.imshow("mov", frame)
    # 次フレーム読み込み処理
    ret, frame = cap.read()
    frameCount += 1

cv2.destroyAllWindows()
cap.release()

orbits.to_csv("output/orbits.csv")
