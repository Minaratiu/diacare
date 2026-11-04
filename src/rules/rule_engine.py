
class RuleEngine:
    """Kiểm tra tính hợp lệ và quy tắc an toàn cơ bản."""

    REQUIRED_FEATURES = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    def validate_assessment(self, health_data):
        if not isinstance(health_data, dict):
            return False, "Du lieu suc khoe phai la dictionary."

        missing = [
            key for key in self.REQUIRED_FEATURES
            if key not in health_data
        ]

        if missing:
            return False, (
                "Thieu cac truong: " + ", ".join(missing)
            )

        # Kiểm tra kiểu dữ liệu, giá trị âm và số không hữu hạn
        import math

        for key in self.REQUIRED_FEATURES:
            value = health_data[key]

            if isinstance(value, bool):
                return False, f"{key} khong hop le."

            try:
                number = float(value)
            except (TypeError, ValueError):
                return False, f"{key} phai la gia tri so."

            if not math.isfinite(number) or number < 0:
                return False, f"{key} phai la so huu han va khong am."

        return True, "Du lieu hop le."

    def validate_question(self, question):
        if not isinstance(question, str) or not question.strip():
            return False, "Vui long nhap cau hoi."

        if len(question.strip()) > 2000:
            return False, "Cau hoi qua dai. Vui long rut gon."

        return True, "Cau hoi hop le."
