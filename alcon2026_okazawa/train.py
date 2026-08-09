import cv2
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier


# =========================
# 設定
# =========================

LEARNING_DIR = "learning"
MODEL_DIR = "model"

MAX_SAMPLES = 10000


# クラス番号
RICE = 0
WEED = 1
SIMILAR_WEED = 2


# =========================
# HSVで植物候補を抽出
# =========================

def create_plant_mask(img):

    # BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSVを分離
    h, s, v = cv2.split(hsv)

    # 植物候補
    #
    # H : 緑を中心にある程度広く取る
    # S : 彩度が低いものを除外
    # V : 暗すぎるものを除外
    #
    mask = (
        (h >= 25)
        & (h <= 95)
        & (s >= 40)
        & (v >= 35)
    )

    mask = mask.astype(np.uint8) * 255

    # 小さいノイズを除去
    kernel = np.ones((5, 5), np.uint8)

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
# 特徴量を作成
# =========================

def create_features(img):

    # BGR
    b, g, r = cv2.split(img)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

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
# 学習画像を読み込む
# =========================

def load_images(folder, label):

    all_features = []

    if not os.path.exists(folder):

        print("フォルダがありません :", folder)

        return (
            np.empty((0, 10)),
            np.empty((0,))
        )

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(
            folder,
            filename
        )

        img = cv2.imread(path)

        if img is None:

            print(
                "画像を読み込めませんでした :",
                path
            )

            continue

        # =========================
        # HSVで植物候補を抽出
        # =========================

        plant_mask = create_plant_mask(img)

        # 特徴量
        features = create_features(img)

        # 植物部分だけ取り出す
        mask = plant_mask.flatten() > 0

        features = features[mask]

        # サンプル数を制限
        if len(features) > MAX_SAMPLES:

            indexes = np.random.choice(
                len(features),
                MAX_SAMPLES,
                replace=False
            )

            features = features[indexes]

        if len(features) == 0:

            print(
                "植物領域がありません :",
                path
            )

            continue

        all_features.append(features)

        print(
            "学習画像 :",
            path,
            "植物画素 :",
            len(features)
        )

    if len(all_features) == 0:

        return (
            np.empty((0, 10)),
            np.empty((0,))
        )

    x = np.vstack(all_features)

    y = np.full(
        len(x),
        label
    )

    return x, y


# =========================
# 学習
# =========================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    all_x = []
    all_y = []


    # -------------------------
    # 水稲
    # -------------------------

    x, y = load_images(
        os.path.join(
            LEARNING_DIR,
            "rice"
        ),
        RICE
    )

    all_x.append(x)
    all_y.append(y)


    # -------------------------
    # 雑草
    # -------------------------

    x, y = load_images(
        os.path.join(
            LEARNING_DIR,
            "weed"
        ),
        WEED
    )

    all_x.append(x)
    all_y.append(y)


    # -------------------------
    # 水稲に似た雑草
    # -------------------------

    x, y = load_images(
        os.path.join(
            LEARNING_DIR,
            "similar_weed"
        ),
        SIMILAR_WEED
    )

    all_x.append(x)
    all_y.append(y)


    # -------------------------
    # データ結合
    # -------------------------

    x = np.vstack(all_x)
    y = np.concatenate(all_y)


    print()
    print("============================")
    print("学習データ")
    print("============================")

    print(
        "総サンプル数 :",
        len(x)
    )

    print(
        "水稲 :",
        np.sum(y == RICE)
    )

    print(
        "雑草 :",
        np.sum(y == WEED)
    )

    print(
        "類似雑草 :",
        np.sum(y == SIMILAR_WEED)
    )


    # -------------------------
    # Random Forest
    # -------------------------

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )


    print()
    print("学習開始...")


    model.fit(
        x,
        y
    )


    print("学習終了")


    # -------------------------
    # モデル保存
    # -------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "rice_weed_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )


    print()
    print(
        "モデルを保存しました :",
        model_path
    )


if __name__ == "__main__":
    main()