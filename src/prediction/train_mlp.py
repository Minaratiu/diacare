
from pathlib import Path
import random

import joblib
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# 1. CẤU HÌNH VÀ ĐƯỜNG DẪN
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.npz"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.npz"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "mlp_diabetes.pth"
RESULT_PATH = MODEL_DIR / "training_results.txt"

RANDOM_STATE = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SIZE = 0.2


# ==================================================
# 2. CỐ ĐỊNH RANDOM SEED
# ==================================================

def set_seed(seed):
    """Cố định seed để tăng khả năng tái lập kết quả."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# 3. XÂY DỰNG MÔ HÌNH MLP
# ==================================================

class DiabetesMLP(nn.Module):
    """Mạng MLP phục vụ phân loại nguy cơ tiểu đường."""

    def __init__(self, input_size=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==================================================
# 4. ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ
# ==================================================

def load_processed_data():
    """Đọc dữ liệu Train và Test."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay: {TEST_PATH}"
        )

    train_data = np.load(TRAIN_PATH)
    test_data = np.load(TEST_PATH)

    X_train = train_data["X"]
    y_train = train_data["y"]

    X_test = test_data["X"]
    y_test = test_data["y"]

    # Kiểm tra số đặc trưng
    if X_train.shape[1] != 8 or X_test.shape[1] != 8:
        raise ValueError(
            "Du lieu dau vao phai co dung 8 dac trung."
        )

    if len(X_train) != len(y_train):
        raise ValueError("Kich thuoc Train khong khop.")

    if len(X_test) != len(y_test):
        raise ValueError("Kich thuoc Test khong khop.")

    if not np.isfinite(X_train).all():
        raise ValueError("Train chua NaN hoac gia tri vo han.")

    if not np.isfinite(X_test).all():
        raise ValueError("Test chua NaN hoac gia tri vo han.")

    return X_train, y_train, X_test, y_test


# ==================================================
# 5. CHIA TRAIN THÀNH TRAIN VÀ VALIDATION
# ==================================================

def split_validation(X_train, y_train):
    """Tách 20% Train ban đầu làm Validation."""

    X_train_new, X_val, y_train_new, y_val = (
        train_test_split(
            X_train,
            y_train,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train
        )
    )

    return X_train_new, X_val, y_train_new, y_val


# ==================================================
# 6. TẠO DATALOADER
# ==================================================

def create_dataloader(X, y, shuffle):
    """Chuyển dữ liệu NumPy thành PyTorch DataLoader."""

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    ).reshape(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


# ==================================================
# 7. HUẤN LUYỆN MỘT EPOCH
# ==================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """Huấn luyện mô hình qua một epoch."""

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Xóa gradient cũ
        optimizer.zero_grad()

        # Dự đoán đầu ra dạng logits
        outputs = model(X_batch)

        # Tính hàm mất mát
        loss = criterion(outputs, y_batch)

        # Lan truyền ngược
        loss.backward()

        # Cập nhật trọng số
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


# ==================================================
# 8. ĐÁNH GIÁ LOSS TRÊN VALIDATION
# ==================================================

def evaluate_loss(model, dataloader, criterion, device):
    """Tính loss trên tập Validation."""

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


# ==================================================
# 9. HUẤN LUYỆN MÔ HÌNH
# ==================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device
):
    """Huấn luyện và lưu mô hình tốt nhất theo Validation Loss."""

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_epoch = 0

    print("\n========== BAT DAU HUAN LUYEN ==========")

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )

        # Lưu trạng thái có Validation Loss tốt nhất
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f}"
            )

    # Nạp lại mô hình tốt nhất
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    print("\n========== HUAN LUYEN HOAN TAT ==========")
    print(f"Epoch tot nhat: {best_epoch}")
    print(f"Validation Loss tot nhat: {best_val_loss:.4f}")
    print(f"Da luu mo hinh: {MODEL_PATH}")

    return model, best_epoch, best_val_loss


# ==================================================
# 10. ĐÁNH GIÁ TRÊN TẬP TEST
# ==================================================

def evaluate_model(model, X_test, y_test, device):
    """Đánh giá mô hình bằng các chỉ số phân loại."""

    model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        logits = model(X_tensor)

        # Chuyển logits thành score trong khoảng 0 đến 1
        probabilities = torch.sigmoid(logits)

    y_score = probabilities.cpu().numpy().ravel()

    # Ngưỡng 0.5 là ngưỡng phân loại mặc định ban đầu
    y_pred = (y_score >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_score)

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Khong phat hien", "Phat hien"],
        zero_division=0
    )

    print("\n========== KET QUA DANH GIA TEST ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": matrix,
        "classification_report": report
    }


# ==================================================
# 11. LƯU KẾT QUẢ ĐÁNH GIÁ
# ==================================================

def save_results(results):
    """Lưu các chỉ số đánh giá ra file văn bản."""

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("DIACARE - MLP TRAINING RESULTS\n")
        file.write("==============================\n\n")

        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]:

            file.write(
                f"{key}: {results[key]:.4f}\n"
            )

        file.write("\nConfusion Matrix:\n")
        file.write(
            np.array2string(
                results["confusion_matrix"]
            )
        )

        file.write("\n\nClassification Report:\n")
        file.write(results["classification_report"])

    print(f"\nDa luu ket qua: {RESULT_PATH}")


# ==================================================
# 12. CHƯƠNG TRÌNH CHÍNH
# ==================================================

def main():

    set_seed(RANDOM_STATE)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Chọn GPU nếu có, nếu không dùng CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("========== DIACARE MLP ==========")
    print(f"Thiet bi huan luyen: {device}")

    # Đọc dữ liệu
    X_train, y_train, X_test, y_test = load_processed_data()

    print(f"\nTrain ban dau: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    # Tách Validation từ Train
    X_train_new, X_val, y_train_new, y_val = split_validation(
        X_train,
        y_train
    )

    print(f"Train sau khi tach: {X_train_new.shape}")
    print(f"Validation: {X_val.shape}")

    # Tạo DataLoader
    train_loader = create_dataloader(
        X_train_new,
        y_train_new,
        shuffle=True
    )

    val_loader = create_dataloader(
        X_val,
        y_val,
        shuffle=False
    )

    # Khởi tạo mô hình
    model = DiabetesMLP(input_size=8).to(device)

    # Huấn luyện
    model, best_epoch, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        device
    )

    # Đánh giá trên Test
    results = evaluate_model(
        model,
        X_test,
        y_test,
        device
    )

    # Lưu kết quả
    save_results(results)

    # Lưu thông tin cấu hình để tái lập
    config = {
        "input_size": 8,
        "hidden_layers": [64, 32],
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "random_state": RANDOM_STATE,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "threshold": 0.5
    }

    joblib.dump(
        config,
        MODEL_DIR / "mlp_config.joblib"
    )

    print("\nHOAN THANH TOAN BO QUA TRINH!")


if __name__ == "__main__":
    main()
