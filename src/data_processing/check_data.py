
from pathlib import Path
import pandas as pd


# ==================================================
# 1. KHAI BÁO ĐƯỜNG DẪN VÀ TÊN CỘT
# ==================================================

# Lấy đường dẫn thư mục gốc DiaCare
BASE_DIR = Path(__file__).resolve().parents[2]

# Đường dẫn file CSV
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"

# Bộ dữ liệu Pima gốc không có dòng tiêu đề
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


# ==================================================
# 2. HÀM ĐỌC VÀ KIỂM TRA DỮ LIỆU
# ==================================================

def load_data():
    """Đọc dữ liệu Pima từ file CSV."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMN_NAMES
    )

    # Kiểm tra số cột thực tế
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError("So cot du lieu khong dung.")

    # Chuyển các cột về kiểu số, báo lỗi nếu dữ liệu không hợp lệ
    for column in COLUMN_NAMES:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


def check_data(df):
    """Hiển thị thông tin tổng quan của dữ liệu."""

    print("\n========== 1. KICH THUOC DU LIEU ==========")
    print(f"So dong: {df.shape[0]}")
    print(f"So cot: {df.shape[1]}")

    print("\n========== 2. TEN CAC COT ==========")
    print(df.columns.tolist())

    print("\n========== 3. 5 DONG DAU TIEN ==========")
    print(df.head())

    print("\n========== 4. KIEU DU LIEU ==========")
    print(df.dtypes)

    print("\n========== 5. DU LIEU THIEU ==========")
    print(df.isnull().sum())

    print("\n========== 6. PHAN BO NHAN OUTCOME ==========")
    print(df["Outcome"].value_counts().sort_index())

    print("\n========== 7. THONG KE MO TA ==========")
    print(df.describe())

    print("\n========== 8. KIEM TRA GIA TRI BANG 0 ==========")

    # Các cột có giá trị 0 cần được xem xét như dữ liệu thiếu
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in zero_columns:
        count = (df[column] == 0).sum()
        print(f"{column}: {count}")

    print("\n========== 9. KIEM TRA NHAN ==========")

    invalid_outcome = ~df["Outcome"].isin([0, 1])

    print("So nhan Outcome khong hop le:",
          invalid_outcome.sum())

    print("\n========== KET THUC KIEM TRA ==========")


# ==================================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==================================================

def main():

    print("BAT DAU KIEM TRA DU LIEU PICMA DIABETES")

    try:
        df = load_data()
        check_data(df)

        print("\nDOC DU LIEU THANH CONG!")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nLOI: {error}")


if __name__ == "__main__":
    main()
