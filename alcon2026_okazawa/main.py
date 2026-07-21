import cv2#画像の読み込みや色変換を行うために必要
import numpy as np #配列を扱うため
from pathlib import Path 
from detector import detect 

# 画像フォルダ
image_dir = Path("img")

# jpg・jpeg・JPG・JPEGをすべて取得
extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]

files = []

for ext in extensions:# 拡張子ごとに画像ファイルを取得
    files.extend(image_dir.glob(ext))

# output画像は除外する↓これないと、output画像も処理されてしまう
files = [f for f in files if "-output" not in f.stem]

for file in files:# 画像ファイルごとに処理

    print("------------------------")
    print("画像:", file.name)

    img = cv2.imread(str(file))

    if img is None:
        print("読み込み失敗")
        continue

    rice, weed, output = detect(img)

    total = rice + weed #植物全体の画素数

    if total == 0:#植物が全くない場合は雑草率を0にする。ratio＝割合
        ratio = 0
    else:
        ratio = weed / total * 100#雑草率を計算

    if ratio < 5:#雑草率に応じて判定
        level = 0
    elif ratio < 15:
        level = 1
    elif ratio < 30:
        level = 2
    else:
        level = 3

    save_name = image_dir / f"{file.stem}-output.JPG"

    cv2.imwrite(str(save_name), output) # 出力画像を保存

    print("水稲画素 :", rice)
    print("雑草画素 :", weed)
    print("雑草率 :", round(ratio,2), "%")
    print("判定 :", level)

    cv2.imshow("Result", output)
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
