import cv2
import numpy as np
import joblib


MODEL_PATH = "model/rice_weed_model.pkl"


# クラス番号
RICE = 0
WEED = 1
SIMILAR_WEED = 2


# 学習済みモデル
model = joblib.load(
    MODEL_PATH
)


# =========================
# HSVで植物候補を抽出
# =========================

def create_plant_mask(img):

    # BGR → HSV
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)


    # 植物候補
    mask = (
        (h >= 25)
        & (h <= 95)
        & (s >= 40)
        & (v >= 35)
    )


    mask = mask.astype(
        np.uint8
    ) * 255


    # ノイズ除去
    kernel = np.ones(
        (5, 5),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    # 穴を埋める
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    return mask


# =========================
# 特徴量
# =========================

def create_features(img):

    b, g, r = cv2.split(img)


    # HSV
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)


    # Lab
    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, lab_b = cv2.split(lab)


    # ExG
    exg = (
        2 * g.astype(np.float32)
        - r.astype(np.float32)
        - b.astype(np.float32)
    )


    features = np.column_stack([

        b.flatten(),
        g.flatten(),
        r.flatten(),

        h.flatten(),
        s.flatten(),
        v.flatten(),

        l.flatten(),
        a.flatten(),
        lab_b.flatten(),

        exg.flatten()

    ])


    return features


# =========================
# 画像判定
# =========================

def detect(img):

    height, width = img.shape[:2]


    # -------------------------
    # ① HSVで植物候補を抽出
    # -------------------------

    plant_mask = create_plant_mask(
        img
    )


    # -------------------------
    # ② 特徴量作成
    # -------------------------

    features = create_features(
        img
    )


    # -------------------------
    # ③ 機械学習
    # -------------------------

    prediction = model.predict(
        features
    )


    prediction = prediction.reshape(
        height,
        width
    )


    # -------------------------
    # HSVで除外された部分
    # -------------------------

    background_mask = (
        plant_mask == 0
    )


    # -------------------------
    # 水稲
    # -------------------------

    rice_mask = (
        (plant_mask > 0)
        &
        (prediction == RICE)
    )


    # -------------------------
    # 雑草
    #
    # weed + similar_weed
    # -------------------------

    weed_mask = (
        (plant_mask > 0)
        &
        (
            (prediction == WEED)
            |
            (prediction == SIMILAR_WEED)
        )
    )


    # -------------------------
    # 画素数
    # -------------------------

    rice_pixel = np.count_nonzero(
        rice_mask
    )


    weed_pixel = np.count_nonzero(
        weed_mask
    )


    background_pixel = np.count_nonzero(
        background_mask
    )


    # -------------------------
    # 結果画像
    # -------------------------

    result = np.zeros_like(
        img
    )


    # 水稲 → 灰色
    result[rice_mask] = (
        150,
        150,
        150
    )


    # 雑草 → 白
    result[weed_mask] = (
        255,
        255,
        255
    )


    # 背景 → 黒
    result[background_mask] = (
        0,
        0,
        0
    )


    return (
        rice_pixel,
        weed_pixel,
        background_pixel,
        result
    )