class RiskAssessmentAgent:
    """
    Đánh giá các chỉ số sức khỏe dựa trên ngưỡng tham chiếu y tế.

    Không chẩn đoán bệnh và không tạo xác suất mắc tiểu đường.
    """

    SOURCES = [
        {
            "name": "ADA - Standards of Care in Diabetes 2026",
            "url": "https://diabetesjournals.org/care/article/49/Supplement_1/S27/163926"
        },
        {
            "name": "NIDDK - Diabetes Tests & Diagnosis",
            "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis"
        },
        {
            "name": "WHO Expert Consultation - BMI in Asian populations",
            "url": "https://doi.org/10.1016/S0140-6736(03)15268-3"
        }
    ]

    def _assess_glucose(self, glucose, test_type):
        """Đối chiếu glucose theo đúng loại xét nghiệm."""

        if test_type == "fasting":
            if glucose < 100:
                category = "Trong ngưỡng bình thường"
                explanation = (
                    "Glucose lúc đói dưới 100 mg/dL."
                )
            elif glucose < 126:
                category = "Nằm trong vùng tiền tiểu đường"
                explanation = (
                    "Glucose lúc đói từ 100 đến 125 mg/dL."
                )
            else:
                category = "Đạt ngưỡng xét nghiệm tiểu đường"
                explanation = (
                    "Glucose lúc đói từ 126 mg/dL trở lên. "
                    "Cần được nhân viên y tế đánh giá và xác nhận."
                )

        elif test_type == "ogtt_2h":
            if glucose < 140:
                category = "Trong ngưỡng bình thường"
                explanation = (
                    "Glucose sau 2 giờ OGTT dưới 140 mg/dL."
                )
            elif glucose < 200:
                category = "Nằm trong vùng rối loạn dung nạp glucose"
                explanation = (
                    "Glucose sau 2 giờ OGTT từ 140 đến 199 mg/dL."
                )
            else:
                category = "Đạt ngưỡng xét nghiệm tiểu đường"
                explanation = (
                    "Glucose sau 2 giờ OGTT từ 200 mg/dL trở lên. "
                    "Cần được nhân viên y tế đánh giá và xác nhận."
                )

        else:
            return {
                "name": "Glucose",
                "value": glucose,
                "unit": "mg/dL",
                "status": "Chưa đủ thông tin",
                "interpretation": (
                    "Chưa xác định loại xét nghiệm glucose. "
                    "Không thể áp dụng ngưỡng lúc đói hoặc OGTT "
                    "khi chưa biết cách đo."
                )
            }

        return {
            "name": "Glucose",
            "value": glucose,
            "unit": "mg/dL",
            "status": category,
            "interpretation": explanation
        }

    def _assess_bmi(self, bmi, age):
        """Đánh giá BMI theo các mốc hành động sức khỏe ở người châu Á."""

        if age < 18:
            return {
                "name": "BMI",
                "value": bmi,
                "unit": "kg/m²",
                "status": "Chưa thể áp dụng ngưỡng người trưởng thành",
                "interpretation": (
                    "Người dưới 18 tuổi cần đánh giá BMI theo tuổi "
                    "và giới tính, không sử dụng trực tiếp ngưỡng "
                    "BMI dành cho người trưởng thành."
                )
            }

        if bmi < 18.5:
            status = "BMI thấp"
            interpretation = (
                "BMI dưới 18,5 kg/m². Đây là mức BMI thấp "
                "theo phân loại người trưởng thành."
            )

        elif bmi < 23:
            status = "Chưa chạm mốc hành động tăng nguy cơ"
            interpretation = (
                "BMI nằm dưới mốc hành động 23 kg/m² "
                "được WHO đề xuất cho quần thể châu Á."
            )

        elif bmi < 27.5:
            status = "Mốc hành động tăng nguy cơ"
            interpretation = (
                "BMI từ 23 đến dưới 27,5 kg/m². "
                "WHO xác định đây là khoảng cần chú ý "
                "về nguy cơ sức khỏe ở quần thể châu Á."
            )

        else:
            status = "Mốc hành động nguy cơ cao"
            interpretation = (
                "BMI từ 27,5 kg/m² trở lên. "
                "WHO xác định đây là mốc hành động nguy cơ cao "
                "đối với sức khỏe ở quần thể châu Á."
            )

        return {
            "name": "BMI",
            "value": bmi,
            "unit": "kg/m²",
            "status": status,
            "interpretation": interpretation
        }

    def assess(self, health_data, glucose_test_type=None):
        """
        Đánh giá dữ liệu sức khỏe.

        glucose_test_type:
            - fasting: glucose lúc đói
            - ogtt_2h: glucose sau 2 giờ OGTT
            - None: chưa xác định loại xét nghiệm
        """

        if not isinstance(health_data, dict):
            return {
                "success": False,
                "message": "Dữ liệu sức khỏe không hợp lệ."
            }

        required_fields = [
            "Glucose",
            "BMI",
            "Age"
        ]

        missing = [
            field for field in required_fields
            if field not in health_data
        ]

        if missing:
            return {
                "success": False,
                "message": f"Thiếu dữ liệu: {missing}"
            }

        try:
            glucose = float(health_data["Glucose"])
            bmi = float(health_data["BMI"])
            age = int(health_data["Age"])

            if glucose < 0 or bmi < 0 or age < 1:
                raise ValueError("Giá trị không hợp lệ.")

        except (TypeError, ValueError, OverflowError):
            return {
                "success": False,
                "message": "Các chỉ số sức khỏe không hợp lệ."
            }

        if glucose_test_type not in (None, "fasting", "ogtt_2h"):
            return {
                "success": False,
                "message": "Loại xét nghiệm glucose không hợp lệ."
            }

        glucose_result = self._assess_glucose(
            glucose,
            glucose_test_type
        )

        bmi_result = self._assess_bmi(
            bmi,
            age
        )

        findings = [
            glucose_result,
            bmi_result
        ]

        # Không cộng điểm hoặc tự tạo mức nguy cơ tổng hợp.
        if glucose_result["status"] == "Đạt ngưỡng xét nghiệm tiểu đường":
            conclusion = (
                "Chỉ số glucose đạt ngưỡng xét nghiệm được sử dụng "
                "trong đánh giá tiểu đường. Đây chưa phải kết luận "
                "chẩn đoán. Bạn nên trao đổi với nhân viên y tế "
                "để thực hiện xét nghiệm xác nhận."
            )

        elif glucose_result["status"] in (
            "Nằm trong vùng tiền tiểu đường",
            "Nằm trong vùng rối loạn dung nạp glucose"
        ):
            conclusion = (
                "Chỉ số glucose nằm trong vùng cần chú ý. "
                "Bạn nên trao đổi với nhân viên y tế để được "
                "đánh giá và theo dõi phù hợp."
            )

        elif bmi_result["status"] in (
            "Mốc hành động tăng nguy cơ",
            "Mốc hành động nguy cơ cao"
        ):
            conclusion = (
                "BMI đã chạm mốc hành động sức khỏe được đề xuất "
                "cho người châu Á. Đây là yếu tố cần chú ý, "
                "không đồng nghĩa với việc mắc bệnh tiểu đường."
            )

        else:
            conclusion = (
                "Chưa ghi nhận chỉ số nào trong các dữ liệu được "
                "đánh giá chạm ngưỡng cần chú ý đã áp dụng. "
                "Chưa đủ căn cứ để kết luận nguy cơ tiểu đường "
                "tổng thể thấp hoặc loại trừ bệnh."
            )

        return {
            "success": True,
            "title": "ĐÁNH GIÁ CÁC CHỈ SỐ SỨC KHỎE",
            "conclusion": conclusion,
            "findings": findings,
            "limitations": [
                "Đánh giá glucose phụ thuộc vào loại xét nghiệm.",
                "BMI chỉ là một chỉ số tham khảo, không chẩn đoán tiểu đường.",
                "Huyết áp hiện chỉ có chỉ số tâm trương.",
                "Số lần mang thai không cho biết người dùng từng mắc "
                "tiểu đường thai kỳ hay không.",
                "Insulin, SkinThickness và DiabetesPedigreeFunction "
                "được giữ làm dữ liệu đầu vào MLP, không được dùng "
                "làm ngưỡng chẩn đoán độc lập.",
                "Chưa có HbA1c, tiền sử gia đình, vòng eo, "
                "tiền sử tiểu đường thai kỳ và các thông tin "
                "cần thiết khác để đánh giá toàn diện."
            ],
            "recommendations": [
                "Nếu chưa có xét nghiệm phù hợp, hãy trao đổi với "
                "nhân viên y tế về xét nghiệm HbA1c hoặc glucose "
                "huyết tương lúc đói.",
                "Nếu kết quả xét nghiệm đạt ngưỡng tiểu đường, "
                "cần được đánh giá và xác nhận theo hướng dẫn chuyên môn.",
                "Duy trì chế độ ăn uống cân đối, hoạt động thể chất "
                "phù hợp và theo dõi sức khỏe định kỳ."
            ],
            "sources": self.SOURCES,
            "disclaimer": (
                "Đây là đánh giá tham khảo dựa trên các ngưỡng "
                "và dữ liệu được cung cấp, không phải chẩn đoán "
                "hoặc thay thế tư vấn y tế chuyên môn."
            )
        }