import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


class MedicalReportAgent:
    """
    Tạo báo cáo giải thích kết quả MLP và Integrated Gradients.

    Gemini chỉ giải thích kết quả mô hình, không chẩn đoán y khoa.
    """

    SYSTEM_INSTRUCTION = """
Bạn là trợ lý giải thích kết quả đánh giá nguy cơ sức khỏe của DiaCare.

Hãy viết báo cáo bằng tiếng Việt tự nhiên, dễ hiểu, chuyên nghiệp
và đồng cảm. Xưng hô với người đọc là "bạn".

QUY TẮC:
- Chỉ sử dụng dữ liệu được cung cấp.
- Không chẩn đoán người dùng mắc hoặc không mắc tiểu đường.
- Không gọi model_score là xác suất mắc bệnh.
- Giải thích model_score và threshold là đầu ra và ngưỡng
  phân loại của mô hình, chỉ mang tính tham khảo.
- Integrated Gradients thể hiện mức đóng góp của đặc trưng
  vào đầu ra mô hình so với baseline, không chứng minh quan hệ
  nhân quả và không thể hiện mức độ nguy hiểm y khoa.
- Đóng góp dương đẩy đầu ra mô hình lên; đóng góp âm kéo
  đầu ra mô hình xuống.
- Không tự suy diễn nguyên nhân gây bệnh.
- Không tự đặt ngưỡng chẩn đoán hoặc bịa thông tin.
- Nêu rõ giới hạn của mô hình và dữ liệu.
- Khuyến khích trao đổi với nhân viên y tế khi cần.

BÁO CÁO CẦN CÓ:
A. Nhận xét tổng quan.
B. Phân tích các yếu tố có đóng góp đáng kể.
C. Nhận xét về dữ liệu và giới hạn.
D. Khuyến nghị chung, an toàn.
E. Lưu ý kết quả không thay thế đánh giá y tế chuyên môn.

Không trả lời bằng checklist Yes/No.
Không giải thích cách bạn tuân thủ chỉ dẫn.
Hãy viết trực tiếp báo cáo dành cho người dùng.
"""

    RETRYABLE_STATUS_CODES = {
        408, 429, 500, 502, 503, 504
    }

    def __init__(self):
        load_dotenv(override=True)

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Chưa cấu hình GEMINI_API_KEY trong file .env"
            )

        self.model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash"
        )

        # Có thể ghi đè danh sách dự phòng trong .env.
        fallback_setting = os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-3.5-flash,gemini-2.5-flash"
        )

        self.fallback_models = [
            model.strip()
            for model in fallback_setting.split(",")
            if model.strip() and model.strip() != self.model_name
        ]

        # Giới hạn thời gian chờ mỗi request: 30 giây.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30_000
            )
        )

    @classmethod
    def _is_retryable_error(cls, error):
        """Xác định lỗi có khả năng tạm thời hay không."""

        status_code = (
            getattr(error, "code", None)
            or getattr(error, "status_code", None)
        )

        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None

        if status_code in cls.RETRYABLE_STATUS_CODES:
            return True

        error_text = str(error).lower()

        temporary_keywords = (
            "503",
            "unavailable",
            "high demand",
            "429",
            "resource exhausted",
            "rate limit",
            "timeout",
            "timed out",
            "connection reset",
            "temporarily",
        )

        return any(
            keyword in error_text
            for keyword in temporary_keywords
        )

    def _build_prompt(self, payload):
        return (
            "Hãy viết trực tiếp một báo cáo giải thích kết quả "
            "đánh giá nguy cơ sức khỏe cho người dùng DiaCare.\n\n"
            "DỮ LIỆU ĐẦU VÀO:\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2
            )
            + "\n\n"
            "YÊU CẦU:\n"
            "- Bắt đầu bằng tiêu đề: BÁO CÁO ĐÁNH GIÁ NGUY CƠ SỨC KHỎE.\n"
            "- Phân tích nhóm phân loại, model_score và threshold "
            "bằng đúng giá trị được cung cấp.\n"
            "- Giải thích các đặc trưng Integrated Gradients "
            "có đóng góp đáng kể, đúng chiều dương hoặc âm.\n"
            "- Nêu giới hạn của kết quả và dữ liệu.\n"
            "- Đưa ra khuyến nghị chung, an toàn.\n"
            "- Viết thành các đoạn văn hoàn chỉnh, có tiêu đề.\n"
            "- Không bịa thông tin, không chẩn đoán và không gọi "
            "model_score là xác suất mắc bệnh.\n"
        )

    def _generate_with_model(self, model_name, prompt):
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=2500
            )
        )

        report = response.text

        if not report or not report.strip():
            raise RuntimeError(
                f"Model {model_name} không trả về nội dung."
            )

        return report.strip()

    def generate_report(self, health_data, prediction):
        """Tạo báo cáo từ dữ liệu sức khỏe và kết quả PredictionAgent."""

        if not isinstance(health_data, dict):
            return {
                "success": False,
                "message": "Dữ liệu sức khỏe không hợp lệ.",
                "report": None
            }

        if not isinstance(prediction, dict):
            return {
                "success": False,
                "message": "Kết quả dự đoán không hợp lệ.",
                "report": None
            }

        if prediction.get("success") is not True:
            return {
                "success": False,
                "message": "PredictionAgent chưa tạo được kết quả.",
                "report": None
            }

        payload = {
            "health_data": health_data,
            "prediction": {
                "predicted_class": prediction.get("predicted_class"),
                "label": prediction.get("label"),
                "model_score": prediction.get("model_score"),
                "threshold": prediction.get("threshold"),
                "explanation": prediction.get("explanation")
            }
        }

        prompt = self._build_prompt(payload)

        # Model chính được thử tối đa 2 lần.
        # Model dự phòng mỗi model được thử 1 lần.
        model_attempts = [
            (self.model_name, 2)
        ] + [
            (model, 1)
            for model in self.fallback_models
        ]

        errors = []

        for model_name, max_attempts in model_attempts:
            for attempt in range(1, max_attempts + 1):
                try:
                    report = self._generate_with_model(
                        model_name,
                        prompt
                    )

                    return {
                        "success": True,
                        "message": "Tạo báo cáo thành công.",
                        "report": report,
                        "model": model_name
                    }

                except Exception as error:
                    errors.append(
                        f"{model_name}, lần {attempt}: {error}"
                    )

                    retryable = self._is_retryable_error(error)

                    # Lỗi xác thực hoặc lỗi không thể thử lại:
                    # không tiếp tục gọi các model khác.
                    if not retryable:
                        return {
                            "success": False,
                            "message": (
                                f"Không thể tạo báo cáo Gemini: {error}"
                            ),
                            "report": None
                        }

                    # Chờ 2 giây trước lần thử lại.
                    if attempt < max_attempts:
                        time.sleep(2)

        error_summary = "\n".join(errors)

        return {
            "success": False,
            "message": (
                "Tất cả model Gemini đều không khả dụng "
                "sau khi thử lại.\n"
                + error_summary
            ),
            "report": None
        }

    def close(self):
        if self.client is not None:
            self.client.close()