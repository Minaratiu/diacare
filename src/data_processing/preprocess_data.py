
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. KHAI BÁO ĐƯỜNG DẪN VÀ CẤU HÌNH
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

FEATURE_COLUMNS = COLUMN_NAMES[:-1]
TARGET_COLUMN = "Outcome"

# Các cột có giá trị 0 cần xem là dữ liệu thiếu
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==================================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU
# ==================================================

def load_data():
    """Đọc dữ liệu Pima từ file CSV."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=COLUMN_NAMES
    )

    if df.shape[1] != 9:
        raise ValueError("Du lieu phai co dung 9 cot.")

    if df.empty:
        raise ValueError("File du lieu khong co ban ghi.")

    # Chuyển dữ liệu về dạng số
    for column in COLUMN_NAMES:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    # Kiểm tra nhãn đầu ra
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "Cot Outcome chi duoc phep chua gia tri 0 hoac 1."
        )

    # Kiểm tra giá trị vô hạn
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Du lieu chua gia tri vo han.")

    return df


# ==================================================
# 3. XỬ LÝ CÁC GIÁ TRỊ 0 BẤT THƯỜNG
# ==================================================

def handle_invalid_zeros(df):
    """Chuyển giá trị 0 bất thường thành NaN."""

    df = df.copy()

    print("\n========== KIEM TRA GIA TRI 0 ==========")

    for column in ZERO_AS_MISSING_COLUMNS:

        zero_count = (df[column] == 0).sum()

        print(f"{column}: {zero_count} gia tri 0")

        # Chỉ xử lý 5 cột đã xác định
        df.loc[df[column] == 0, column] = np.nan

    return df


# ==================================================
# 4. CHIA TẬP TRAIN VÀ TEST
# ==================================================

def split_data(df):
    """Chia dữ liệu theo tỷ lệ 80% Train và 20% Test."""

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ==================================================
# 5. ĐIỀN GIÁ TRỊ THIẾU VÀ CHUẨN HÓA
# ==================================================

def preprocess_features(X_train, X_test):
    """
    Điền giá trị thiếu bằng median và chuẩn hóa dữ liệu.

    Chỉ fit trên tập Train để tránh Data Leakage.
    """

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Học median và tham số chuẩn hóa từ Train
    X_train_processed = preprocessor.fit_transform(X_train)

    # Chỉ áp dụng tham số đã học lên Test
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed.astype(np.float32),
        X_test_processed.astype(np.float32),
        preprocessor
    )


# ==================================================
# 6. LƯU DỮ LIỆU VÀ BỘ TIỀN XỬ LÝ
# ==================================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    """Lưu dữ liệu đã xử lý và bộ tiền xử lý."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Lưu tập Train
    np.savez_compressed(
        PROCESSED_DIR / "train.npz",
        X=X_train,
        y=y_train.to_numpy(dtype=np.float32),
        feature_names=np.array(FEATURE_COLUMNS)
    )

    # Lưu tập Test
    np.savez_compressed(
        PROCESSED_DIR / "test.npz",
        X=X_test,
        y=y_test.to_numpy(dtype=np.float32),
        feature_names=np.array(FEATURE_COLUMNS)
    )

    # Lưu bộ tiền xử lý để sử dụng khi dự đoán dữ liệu mới
    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.joblib"
    )

    print("\n========== DA LUU FILE ==========")
    print("data/processed/train.npz")
    print("data/processed/test.npz")
    print("models/preprocessor.joblib")


# ==================================================
# 7. HIỂN THỊ KẾT QUẢ
# ==================================================

def display_results(
    X_train,
    X_test,
    y_train,
    y_test
):
    """Hiển thị thông tin sau tiền xử lý."""

    print("\n========== KET QUA TIEN XU LY ==========")

    print(f"Train: {X_train.shape[0]} dong")
    print(f"Test : {X_test.shape[0]} dong")

    print(f"So dac trung: {X_train.shape[1]}")

    print("\nPhan bo nhan Train:")
    print(y_train.value_counts().sort_index())

    print("\nPhan bo nhan Test:")
    print(y_test.value_counts().sort_index())

    print("\nKiem tra NaN trong Train:")
    print(np.isnan(X_train).sum())

    print("\nKiem tra NaN trong Test:")
    print(np.isnan(X_test).sum())

    print("\nKich thuoc mang Train:", X_train.shape)
    print("Kich thuoc mang Test :", X_test.shape)


# ==================================================
# 8. CHƯƠNG TRÌNH CHÍNH
# ==================================================

def main():

    print("BAT DAU TIEN XU LY DU LIEU DIABETES")

    # Bước 1: Đọc dữ liệu
    df = load_data()

    print(f"\nSo dong du lieu goc: {len(df)}")

    # Bước 2: Xử lý giá trị 0
    df = handle_invalid_zeros(df)

    # Bước 3: Chia Train và Test
    X_train, X_test, y_train, y_test = split_data(df)

    # Bước 4: Điền dữ liệu thiếu và chuẩn hóa
    X_train_processed, X_test_processed, preprocessor = (
        preprocess_features(X_train, X_test)
    )

    # Bước 5: Lưu kết quả
    save_processed_data(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )

    # Bước 6: Hiển thị kết quả
    display_results(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )

    print("\nHOAN THANH TIEN XU LY DU LIEU!")


if __name__ == "__main__":
    main()
