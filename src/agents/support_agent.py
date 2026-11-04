
class SupportAgent:
    """Tổng hợp kết quả từ các Agent thành phản hồi cho người dùng."""

    DISCLAIMER = (
        "Lưu ý: Kết quả chỉ mang tính tham khảo, "
        "không thay thế chẩn đoán hoặc tư vấn của nhân viên y tế."
    )

    def _format_prediction(self, prediction):
        """Trình bày kết quả từ MLP và Integrated Gradients."""

        sections = ["KẾT QUẢ PHÂN TÍCH TỪ MÔ HÌNH MLP"]

        if not prediction.get("success"):
            sections.append(
                prediction.get("message", "Mô hình không thể thực hiện dự đoán.")
            )
            return sections

        sections.append(
            f"Nhóm phân loại: {prediction.get('label', 'Chưa xác định')}"
        )
        sections.append(
            f"Điểm đầu ra mô hình: {prediction.get('model_score', 'N/A')}"
        )
        sections.append(
            f"Ngưỡng phân loại: {prediction.get('threshold', 'N/A')}"
        )
        sections.append(
            "Điểm đầu ra chưa được hiệu chuẩn lâm sàng, "
            "không phải xác suất mắc bệnh."
        )

        explanation = prediction.get("explanation") or {}

        if explanation.get("success"):
            contributions = explanation.get("feature_contributions", [])

            if contributions:
                sections.append("\nCÁC YẾU TỐ ẢNH HƯỞNG ĐẾN ĐẦU RA MÔ HÌNH")

                for item in contributions:
                    direction = item.get("direction", "neutral")

                    if direction == "increases_model_output":
                        meaning = "làm tăng đầu ra mô hình"
                    elif direction == "decreases_model_output":
                        meaning = "làm giảm đầu ra mô hình"
                    else:
                        meaning = "ảnh hưởng không đáng kể"

                    sections.append(
                        f"- {item.get('feature', 'Không xác định')}: "
                        f"{meaning} "
                        f"(đóng góp: {item.get('contribution', 'N/A')})"
                    )

                sections.append(
                    "Lưu ý: Đóng góp Integrated Gradients giải thích "
                    "cách mô hình phản ứng với dữ liệu, không chứng minh "
                    "quan hệ nhân quả hoặc mức độ nguy cơ lâm sàng."
                )

        return sections

    def _format_risk_assessment(self, assessment):
        """Trình bày đánh giá dựa trên ngưỡng tham chiếu y tế."""

        sections = ["\nĐÁNH GIÁ CÁC CHỈ SỐ SỨC KHỎE"]

        if not assessment.get("success"):
            sections.append(
                assessment.get("message", "Không thể thực hiện đánh giá.")
            )
            return sections

        sections.append(
            f"Kết luận tham khảo: {assessment.get('conclusion', 'Chưa có kết luận.')}"
        )

        # Hiển thị kết quả từng chỉ số
        sections.append("\nPHÂN TÍCH TỪNG CHỈ SỐ")

        for finding in assessment.get("findings", []):
            name = finding.get("name", "Chỉ số")
            value = finding.get("value", "N/A")
            unit = finding.get("unit", "")
            status = finding.get("status", "Chưa xác định")
            interpretation = finding.get("interpretation", "")

            sections.append(
                f"\n- {name}: {value} {unit}".strip()
            )
            sections.append(f"  Phân loại: {status}")

            if interpretation:
                sections.append(f"  Giải thích: {interpretation}")

        # Giới hạn dữ liệu
        limitations = assessment.get("limitations", [])

        if limitations:
            sections.append("\nGIỚI HẠN CỦA ĐÁNH GIÁ")

            for limitation in limitations:
                sections.append(f"- {limitation}")

        # Khuyến nghị
        recommendations = assessment.get("recommendations", [])

        if recommendations:
            sections.append("\nKHUYẾN NGHỊ")

            for recommendation in recommendations:
                sections.append(f"- {recommendation}")

        # Nguồn tham khảo
        sources = assessment.get("sources", [])

        if sources:
            sections.append("\nNGUỒN THAM KHẢO")

            for source in sources:
                sections.append(
                    f"- {source.get('name', 'Nguồn y tế')}: "
                    f"{source.get('url', '')}"
                )

        return sections

    def _format_knowledge(self, knowledge):
        """Trình bày kết quả tra cứu tri thức."""

        sections = ["\nKẾT QUẢ TRA CỨU KIẾN THỨC"]

        if not knowledge.get("success"):
            sections.append(
                knowledge.get("message", "Không truy xuất được tài liệu.")
            )
            return sections

        results = knowledge.get("results", [])

        if not results:
            sections.append(
                "Chưa tìm thấy tài liệu đủ liên quan để trả lời."
            )
            return sections

        for index, item in enumerate(results, start=1):
            sections.append(
                f"\nTài liệu {index}: {item.get('content', '')}"
            )

        sources = knowledge.get("sources", [])

        if sources:
            sections.append("\nNGUỒN THAM KHẢO")

            for source in sources:
                sections.append(
                    f"- {source.get('name', 'Nguồn tài liệu')}: "
                    f"{source.get('url', '')}"
                )

        return sections

    def generate(self, agent_results):
        """Tổng hợp kết quả từ các Agent."""

        sections = []

        prediction = agent_results.get("prediction")

        if prediction:
            sections.extend(self._format_prediction(prediction))

        risk_assessment = agent_results.get("risk_assessment")

        if risk_assessment:
            sections.extend(
                self._format_risk_assessment(risk_assessment)
            )

        knowledge = agent_results.get("knowledge")

        if knowledge:
            sections.extend(self._format_knowledge(knowledge))

        sections.append("\n" + self.DISCLAIMER)

        return "\n".join(sections)
