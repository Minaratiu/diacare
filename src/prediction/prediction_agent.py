
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients
from pathlib import Path


# Thứ tự 8 thuộc tính phải giống lúc huấn luyện mô hình
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


class DiabetesMLP(nn.Module):
    """
    Kiến trúc MLP dùng để dự đoán nguy cơ tiểu đường.
    """

    def __init__(self, input_size=8, hidden_layers=None):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PredictionAgent:
    """
    Agent dự đoán nguy cơ tiểu đường bằng MLP.

    Tích hợp Integrated Gradients để giải thích
    mức đóng góp của từng đặc trưng vào đầu ra mô hình.
    """

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[2]

        self.model_path = self.project_root / "models" / "mlp_diabetes.pth"
        self.config_path = self.project_root / "models" / "mlp_config.joblib"
        self.preprocessor_path = (
            self.project_root / "models" / "preprocessor.joblib"
        )

        # Kiểm tra các file cần thiết
        for path in [
            self.model_path,
            self.config_path,
            self.preprocessor_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Khong tim thay file: {path}"
                )

        # Đọc cấu hình mô hình
        self.config = joblib.load(self.config_path)

        # Đọc bộ tiền xử lý đã huấn luyện
        self.preprocessor = joblib.load(self.preprocessor_path)

        input_size = self.config.get("input_size", 8)
        hidden_layers = self.config.get("hidden_layers", [64, 32])

        if input_size != len(FEATURE_NAMES):
            raise ValueError(
                "So luong input trong config khong khop voi FEATURE_NAMES."
            )

        # Khởi tạo kiến trúc MLP
        self.model = DiabetesMLP(
            input_size=input_size,
            hidden_layers=hidden_layers,
        )

        # Nạp trọng số đã huấn luyện
        state_dict = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Ngưỡng phân loại lấy từ cấu hình
        self.threshold = float(
            self.config.get("threshold", 0.5)
        )

    def _forward_score(self, inputs):
        """
        Chuyển logits thành model score trong khoảng 0-1.
        """

        logits = self.model(inputs)

        return torch.sigmoid(logits).squeeze(-1)

    def explain_prediction(self, input_tensor):
        """
        Giải thích đầu ra MLP bằng Integrated Gradients.

        Baseline là vector 0 trong không gian đặc trưng
        sau khi đã StandardScaler.

        Attribution thể hiện mức đóng góp vào đầu ra
        của mô hình so với baseline này.
        """

        self.model.eval()

        baseline = torch.zeros_like(input_tensor)

        integrated_gradients = IntegratedGradients(
            self._forward_score
        )

        attributions, convergence_delta = (
            integrated_gradients.attribute(
                inputs=input_tensor,
                baselines=baseline,
                n_steps=64,
                return_convergence_delta=True,
            )
        )

        # Lấy kết quả attribution
        attribution_values = (
            attributions.detach()
            .cpu()
            .numpy()[0]
        )

        # Điểm đầu ra tại baseline
        with torch.no_grad():
            baseline_score = float(
                self._forward_score(baseline).item()
            )

            output_score = float(
                self._forward_score(input_tensor).item()
            )

        feature_contributions = []

        for name, value in zip(
            FEATURE_NAMES,
            attribution_values,
        ):

            value = float(value)

            if value > 0:
                direction = "increases_model_output"
            elif value < 0:
                direction = "decreases_model_output"
            else:
                direction = "neutral"

            feature_contributions.append({
                "feature": name,
                "contribution": round(value, 6),
                "absolute_contribution": round(abs(value), 6),
                "direction": direction,
            })

        delta_value = float(
            convergence_delta.detach()
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

        return {
            "success": True,
            "method": "Integrated Gradients",
            "baseline": (
                "Zero vector in standardized feature space"
            ),
            "baseline_model_score": round(baseline_score, 6),
            "explained_model_score": round(output_score, 6),
            "feature_contributions": feature_contributions,
            "convergence_delta": round(delta_value, 8),
            "disclaimer": (
                "Cac gia tri dong gop chi giai thich hanh vi "
                "cua mo hinh so voi baseline. Chung khong chung minh "
                "nguyen nhan gay benh va khong phai muc do anh huong "
                "y khoa cua tung yeu to."
            ),
        }

    def predict(self, input_data):
        """
        Dự đoán nguy cơ và giải thích kết quả.

        input_data phải chứa đủ 8 thuộc tính.
        """

        if not isinstance(input_data, dict):
            raise TypeError(
                "input_data phai la dictionary."
            )

        # Kiểm tra các thuộc tính bắt buộc
        missing_features = [
            name
            for name in FEATURE_NAMES
            if name not in input_data
        ]

        if missing_features:
            raise ValueError(
                f"Thieu cac thuoc tinh: {missing_features}"
            )

        values = {}

        for name in FEATURE_NAMES:

            value = input_data[name]

            if isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"{name} khong duoc la gia tri boolean."
                )

            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{name} phai la gia tri so."
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} phai la gia tri huu han."
                )

            if value < 0:
                raise ValueError(
                    f"{name} khong duoc am."
                )

            values[name] = value

        # Đảm bảo đúng thứ tự đặc trưng
        input_df = pd.DataFrame(
            [[values[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )

        # Tiền xử lý bằng preprocessor đã huấn luyện
        processed_data = self.preprocessor.transform(input_df)

        input_tensor = torch.tensor(
            np.asarray(processed_data),
            dtype=torch.float32,
        )

        self.model.eval()

        # Dự đoán bằng MLP
        with torch.no_grad():
            score = float(
                self._forward_score(input_tensor).item()
            )

        predicted_class = int(
            score >= self.threshold
        )

        # Giải thích dự đoán
        try:
            explanation = self.explain_prediction(
                input_tensor
            )

        except Exception as error:
            explanation = {
                "success": False,
                "error": str(error),
            }

        # Kết quả dự đoán
        result = {
            "success": True,
            "predicted_class": predicted_class,

            "label": (
                "Mo hinh phan loai vao nhom nguy co"
                if predicted_class == 1
                else "Mo hinh khong phan loai vao nhom nguy co"
            ),

            "model_score": round(score, 4),
            "threshold": self.threshold,

            "explanation": explanation,

            "disclaimer": (
                "Ket qua chi mang tinh tham khao tu mo hinh "
                "hoc may, khong phai chan doan y khoa. "
                "Model score khong phai xac suat mac benh "
                "duoc hieu chinh lam sang. Nguoi dung nen "
                "tham khao y kien nhan vien y te de duoc "
                "danh gia chinh xac."
            ),
        }

        return result
