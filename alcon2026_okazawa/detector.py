import cv2#画像の読み込みや色変換を行うために必要
import numpy as np#配列を扱うため

def detect(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#画像をBGR形式からHSV形式へ変換
    #H：色相
    #S：彩度
    #V：明度 これにより緑を抽出しやすい
    # 稲
    rice = cv2.inRange(#稲を抽出する処理
        hsv,
        (35,40,40),#稲の最小HSV値
        (90, 255, 255)#稲の最大HSV値
    )

    # 雑草
    weed = cv2.inRange(
        hsv,
        (20,40,40),#雑草の最小HSV値
        (35, 255, 255)#雑草の最大HSV値
    )

    rice_pixel = cv2.countNonZero(rice)#稲のピクセル数をカウント
    weed_pixel = cv2.countNonZero(weed)#雑草のピクセル数をカウント

    output = np.zeros_like(img)#

    output[rice > 0] = (128, 128, 128)#稲をグレーで表示
    output[weed > 0] = (255, 255, 255)#雑草を白で表示

    return weed_pixel, rice_pixel, output