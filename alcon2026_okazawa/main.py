import cv2
import os

from detector import detect


# =========================
# フォルダ
# =========================

INPUT_DIR = "img"
OUTPUT_DIR = "output"


# outputフォルダを作成
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================
# img内の画像を取得
# =========================

files = os.listdir(
    INPUT_DIR
)


count = 0


# =========================
# 全画像を処理
# =========================

for filename in files:

    # すでに作った結果画像は除外
    if "-output" in filename:
        continue


    # 画像ファイルだけ処理
    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue


    input_path = os.path.join(
        INPUT_DIR,
        filename
    )


    # 画像読み込み
    img = cv2.imread(
        input_path
    )


    if img is None:

        print(
            "画像を読み込めませんでした :",
            filename
        )

        continue


    # =========================
    # 判定
    # =========================

    (
        rice_pixel,
        weed_pixel,
        background_pixel,
        result
    ) = detect(img)


    # =========================
    # 雑草率
    # =========================

    plant_pixel = (
        rice_pixel
        + weed_pixel
    )


    if plant_pixel == 0:

        ratio = 0

    else:

        ratio = (
            weed_pixel
            / plant_pixel
            * 100
        )


    # =========================
    # 判定
    # =========================

    if ratio < 8:

        level = 0

    elif ratio < 20:

        level = 1

    elif ratio < 40:

        level = 2

    else:

        level = 3


    # =========================
    # 結果表示
    # =========================

    print(
        "----------------------------"
    )

    print(
        "画像 :",
        filename
    )

    print(
        "水稲画素 :",
        rice_pixel
    )

    print(
        "雑草画素 :",
        weed_pixel
    )

    print(
        "背景画素 :",
        background_pixel
    )

    print(
        "植物画素 :",
        plant_pixel
    )

    print(
        "雑草率 : {:.1f}%".format(
            ratio
        )
    )

    print(
        "判定 :",
        level
    )


    # =========================
    # 結果画像保存
    # =========================

    name, ext = os.path.splitext(
        filename
    )


    output_filename = (
        name
        + "-output"
        + ext
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )


    cv2.imwrite(
        output_path,
        result
    )


    print(
        "結果画像 :",
        output_path
    )


    count += 1


# =========================
# 終了
# =========================

print(
    "----------------------------"
)

print(
    "処理した画像 :",
    count,
    "枚"
)

print(
    "すべての画像の処理が終了しました"
)