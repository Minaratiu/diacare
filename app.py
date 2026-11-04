
import streamlit as st
import pandas as pd


# =========================================================
# 1. CAU HINH UNG DUNG
# =========================================================

st.set_page_config(
    page_title="DiaCare - Ho tro suc khoe",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CSS GIAO DIEN
# =========================================================

st.markdown("""
<style>
    .main-title {
        color: #167D70;
        font-size: 34px;
        font-weight: 750;
        margin-bottom: 0;
    }

    .sub-title {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #167D70;
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-box {
        background-color: #eff8f6;
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #167D70;
        margin: 12px 0;
    }

    .disclaimer {
        background-color: #f1f5f9;
        color: #475569;
        padding: 14px;
        border-radius: 8px;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. KHOI TAO COORDINATOR
# =========================================================

@st.cache_resource(show_spinner="Đang khởi tạo hệ thống DiaCare...")
def get_coordinator():
    from src.orchestration.coordinator import CoordinatorAgent
    return CoordinatorAgent()


def get_system():
    try:
        return get_coordinator()
    except Exception as error:
        st.error("Không thể khởi tạo hệ thống DiaCare.")
        st.exception(error)
        return None


# =========================================================
# 4. CAC HAM HIEN THI KET QUA
# =========================================================

def show_explanation(explanation):
    """
    Hiển thị kết quả giải thích Integrated Gradients.
    """

    st.subheader("Giải thích kết quả mô hình")

    if not explanation:
        st.warning("Chưa có dữ liệu giải thích.")
        return

    if not explanation.get("success"):
        st.warning(
            "Mô hình đã dự đoán nhưng không thể tạo phần giải thích."
        )

        if explanation.get("error"):
            st.caption(f"Chi tiết: {explanation['error']}")

        return

    contributions = explanation.get(
        "feature_contributions", []
    )

    if not contributions:
        st.warning("Không có dữ liệu đóng góp của các đặc trưng.")
        return

    # Chuyển tên thuộc tính kỹ thuật thành tên dễ hiểu
    feature_labels = {
        "Pregnancies": "Số lần mang thai",
        "Glucose": "Glucose",
        "BloodPressure": "Huyết áp tâm trương",
        "SkinThickness": "Độ dày da",
        "Insulin": "Insulin",
        "BMI": "Chỉ số BMI",
        "DiabetesPedigreeFunction": "Yếu tố di truyền",
        "Age": "Tuổi",
    }

    # Sắp xếp theo độ lớn tuyệt đối của mức đóng góp
    sorted_contributions = sorted(
        contributions,
        key=lambda item: abs(
            float(item.get("contribution", 0))
        ),
        reverse=True
    )

    # Tạo bảng dữ liệu hiển thị
    display_data = []

    for item in sorted_contributions:
        feature = item.get("feature", "")
        contribution = float(
            item.get("contribution", 0)
        )

        if contribution > 0:
            direction = "Tăng đầu ra mô hình"
        elif contribution < 0:
            direction = "Giảm đầu ra mô hình"
        else:
            direction = "Gần như trung tính"

        display_data.append({
            "Yếu tố": feature_labels.get(feature, feature),
            "Mức đóng góp": contribution,
            "Chiều tác động": direction
        })

    # Biểu đồ đóng góp
    st.markdown("#### Biểu đồ mức đóng góp của các yếu tố")

    chart_data = pd.DataFrame(display_data)

    chart_data = chart_data.set_index("Yếu tố")

    st.bar_chart(
        chart_data[["Mức đóng góp"]],
        horizontal=True
    )

    st.caption(
        "Giá trị dương: làm tăng đầu ra mô hình. "
        "Giá trị âm: làm giảm đầu ra mô hình. "
        "Độ lớn tuyệt đối thể hiện mức đóng góp theo phương pháp "
        "Integrated Gradients."
    )

    # Bảng chi tiết
    st.markdown("#### Chi tiết đóng góp")

    st.dataframe(
        pd.DataFrame(display_data),
        use_container_width=True,
        hide_index=True
    )

    # So sánh baseline và đầu ra mô hình
    baseline_score = explanation.get(
        "baseline_model_score"
    )

    explained_score = explanation.get(
        "explained_model_score"
    )

    if baseline_score is not None and explained_score is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Đầu ra tại baseline",
                f"{float(baseline_score):.6f}"
            )

        with col2:
            st.metric(
                "Đầu ra tại dữ liệu nhập",
                f"{float(explained_score):.6f}"
            )

    delta = explanation.get("convergence_delta")

    if delta is not None:
        st.caption(
            f"Convergence delta: {float(delta):.8f}"
        )

    with st.expander("Phương pháp giải thích và giới hạn"):
        st.write(
            "**Phương pháp:** Integrated Gradients."
        )

        st.write(
            "**Baseline:** Vector 0 trong không gian đặc trưng "
            "đã chuẩn hóa."
        )

        st.write(
            "Các giá trị đóng góp giải thích cách mô hình MLP "
            "tạo ra đầu ra so với baseline đã chọn. "
            "Chúng không chứng minh nguyên nhân gây bệnh, "
            "không thể hiện mức độ ảnh hưởng y khoa và "
            "không thay thế đánh giá của nhân viên y tế."
        )

    if explanation.get("disclaimer"):
        st.info(explanation["disclaimer"])


def show_prediction(prediction):
    if not prediction:
        return

    st.subheader("Kết quả đánh giá")

    if not prediction.get("success"):
        st.error("Không thể thực hiện đánh giá.")
        return

    predicted_class = prediction.get("predicted_class")
    score = prediction.get("model_score", 0)
    threshold = prediction.get("threshold", 0.5)

    if predicted_class == 1:
        st.warning(
            "Mô hình phân loại dữ liệu đầu vào vào nhóm nguy cơ. "
            "Người dùng nên tham khảo nhân viên y tế để được đánh giá thêm."
        )
    else:
        st.success(
            "Mô hình không phân loại dữ liệu đầu vào vào nhóm nguy cơ. "
            "Kết quả này không loại trừ khả năng mắc bệnh."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Điểm số mô hình",
            value=f"{score:.4f}"
        )

    with col2:
        st.metric(
            label="Ngưỡng phân loại",
            value=f"{threshold:.2f}"
        )

    st.progress(
        max(0.0, min(float(score), 1.0))
    )

    st.caption(
        "Thanh biểu diễn điểm đầu ra của mô hình, không phải xác suất "
        "mắc bệnh đã được hiệu chuẩn lâm sàng."
    )

    # Hiển thị giải thích Integrated Gradients
    show_explanation(
        prediction.get("explanation")
    )

    st.info(prediction.get("disclaimer", ""))


def show_knowledge(knowledge):
    if not knowledge:
        return

    st.subheader("Kết quả tra cứu tri thức")

    if not knowledge.get("success"):
        st.error(
            knowledge.get(
                "message",
                "Không thể truy xuất tri thức."
            )
        )
        return

    st.info(
        knowledge.get(
            "message",
            "Đã hoàn thành truy xuất."
        )
    )

    results = knowledge.get("results", [])

    if not results:
        st.warning(
            "Chưa tìm thấy tài liệu đủ liên quan để trả lời câu hỏi."
        )
        return

    for index, item in enumerate(results, start=1):
        with st.expander(
            f"Tài liệu {index} - Điểm tương đồng: "
            f"{item.get('score', 0):.4f}",
            expanded=(index == 1)
        ):
            st.write(
                item.get("content", "Không có nội dung.")
            )

            st.caption(
                f"Chunk ID: {item.get('chunk_id', 'N/A')}"
            )

            st.caption(
                f"Nguồn: {item.get('source_name') or 'Chưa xác định'}"
            )

            st.caption(
                f"Tổ chức: {item.get('organization') or 'Chưa xác định'}"
            )

    sources = knowledge.get("sources", [])

    if sources:
        st.markdown("#### Nguồn tham khảo")

        for index, source in enumerate(sources, start=1):
            name = source.get("name") or f"Nguồn {index}"
            organization = source.get("organization") or ""
            url = source.get("url")

            if url:
                st.markdown(
                    f"{index}. [{name}]({url})"
                    + (f" - {organization}" if organization else "")
                )
            else:
                st.write(f"{index}. {name}")


def show_agent_result(result):
    if not result:
        return

    if not result.get("success"):
        st.error(
            result.get(
                "message",
                "Hệ thống không thể xử lý yêu cầu."
            )
        )
        return

    st.success("Coordinator đã xử lý yêu cầu.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Trạng thái",
            result.get("state", "N/A")
        )

    with col2:
        st.metric(
            "Hành động Q-Learning",
            result.get("action", "N/A")
        )

    if result.get("prediction") is not None:
        show_prediction(result["prediction"])

    if result.get("knowledge") is not None:
        show_knowledge(result["knowledge"])

    if result.get("response"):
        with st.expander("Nội dung tổng hợp từ Support Agent"):
            st.text(result["response"])


# =========================================================
# 5. THANH DIEU HUONG
# =========================================================

with st.sidebar:
    st.title("🩺 DiaCare")

    st.caption("Hệ thống hỗ trợ sức khỏe")

    st.divider()

    page = st.radio(
        "CHỨC NĂNG",
        [
            "Tổng quan",
            "Đánh giá nguy cơ",
            "Tra cứu tri thức",
            "Hỗ trợ đa Agent"
        ],
        index=0
    )

    st.divider()

    st.markdown("**Công nghệ sử dụng**")
    st.caption("• Deep Learning - MLP")
    st.caption("• Neo4j Knowledge Graph")
    st.caption("• Retrieval-Augmented Generation")
    st.caption("• Q-Learning")
    st.caption("• Rule Engine")

    st.divider()

    st.caption("DiaCare - Phiên bản 1.0")


# =========================================================
# 6. HEADER CHUNG
# =========================================================

st.markdown(
    '<div class="main-title">DiaCare</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Hệ thống hỗ trợ đánh giá nguy cơ và tra cứu tri thức '
    'bệnh tiểu đường ứng dụng trí tuệ nhân tạo'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 7. TRANG TONG QUAN
# =========================================================

if page == "Tổng quan":

    st.markdown(
        '<div class="section-title">Tổng quan hệ thống</div>',
        unsafe_allow_html=True
    )

    st.write(
        "DiaCare hỗ trợ người dùng đánh giá dữ liệu sức khỏe "
        "và tìm kiếm tài liệu tri thức liên quan đến bệnh tiểu đường."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧠")
        st.subheader("Đánh giá nguy cơ")
        st.write(
            "Sử dụng mô hình MLP đã huấn luyện để phân loại "
            "dữ liệu sức khỏe."
        )

    with col2:
        st.markdown("### 📚")
        st.subheader("Tra cứu tri thức")
        st.write(
            "Tìm kiếm các đoạn tài liệu y khoa trong "
            "Knowledge Graph Neo4j."
        )

    with col3:
        st.markdown("### 🤖")
        st.subheader("Hệ đa Agent")
        st.write(
            "Điều phối các Agent thông qua Coordinator "
            "và Q-Learning Router."
        )

    st.divider()

    st.subheader("Hướng dẫn sử dụng")

    st.markdown("""
    1. Chọn **Đánh giá nguy cơ** để nhập 8 chỉ số sức khỏe.
    2. Chọn **Tra cứu tri thức** để đặt câu hỏi về bệnh tiểu đường.
    3. Chọn **Hỗ trợ đa Agent** để gửi yêu cầu kết hợp đánh giá và tra cứu.
    """)

    st.info(
        "Đây là hệ thống hỗ trợ nghiên cứu và tham khảo, "
        "không phải công cụ chẩn đoán bệnh."
    )


# =========================================================
# 8. CHUC NANG DANH GIA NGUY CO
# =========================================================

elif page == "Đánh giá nguy cơ":

    st.markdown(
        '<div class="section-title">Đánh giá nguy cơ tiểu đường</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Nhập các chỉ số sức khỏe để mô hình MLP thực hiện phân loại."
    )

    st.info(
        "Sử dụng số liệu đo thực tế nếu có. Các giá trị mặc định "
        "chỉ phục vụ thử nghiệm giao diện."
    )

    with st.form("assessment_form"):

        st.subheader("Thông tin sức khỏe")

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Số lần mang thai",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )

            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0.0,
                value=120.0,
                step=1.0
            )

            blood_pressure = st.number_input(
                "Blood Pressure - Huyết áp tâm trương (mmHg)",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0.0,
                value=20.0,
                step=1.0
            )

        with col2:

            insulin = st.number_input(
                "Insulin (mu U/mL)",
                min_value=0.0,
                value=80.0,
                step=1.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                value=25.0,
                step=0.1
            )

            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                value=0.5,
                step=0.01
            )

            age = st.number_input(
                "Tuổi",
                min_value=1,
                max_value=120,
                value=30,
                step=1
            )

        submitted = st.form_submit_button(
            "Đánh giá nguy cơ",
            type="primary",
            use_container_width=True
        )

    if submitted:

        health_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }

        system = get_system()

        if system is not None:

            with st.spinner(
                "Đang kiểm tra dữ liệu và chạy mô hình MLP..."
            ):

                try:
                    # Coordinator thực hiện Rule Engine,
                    # Q-Learning Router và Prediction Agent.
                    result = system.process(
                        health_data=health_data
                    )

                    show_agent_result(result)

                except Exception as error:
                    st.error(
                        "Đã xảy ra lỗi khi đánh giá dữ liệu."
                    )
                    st.exception(error)


# =========================================================
# 9. CHUC NANG TRA CUU TRI THUC
# =========================================================

elif page == "Tra cứu tri thức":

    st.markdown(
        '<div class="section-title">Tra cứu tri thức y khoa</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Nhập câu hỏi để tìm kiếm các tài liệu liên quan "
        "trong Knowledge Graph Neo4j."
    )

    with st.form("knowledge_form"):

        question = st.text_area(
            "Câu hỏi của bạn",
            placeholder=(
                "Ví dụ: Những yếu tố nào làm tăng nguy cơ "
                "mắc bệnh tiểu đường type 2?"
            ),
            max_chars=2000,
            height=130
        )

        search_submitted = st.form_submit_button(
            "Tra cứu tri thức",
            type="primary",
            use_container_width=True
        )

    if search_submitted:

        if not question.strip():
            st.warning(
                "Vui lòng nhập câu hỏi trước khi tra cứu."
            )

        else:

            system = get_system()

            if system is not None:

                with st.spinner(
                    "Đang tạo embedding và tìm kiếm tài liệu trong Neo4j..."
                ):

                    try:
                        # S2: chỉ yêu cầu tra cứu tri thức.
                        result = system.process(
                            question=question.strip()
                        )

                        show_agent_result(result)

                    except Exception as error:
                        st.error(
                            "Không thể thực hiện tra cứu tri thức."
                        )
                        st.exception(error)


# =========================================================
# 10. CHUC NANG HO TRO DA AGENT
# =========================================================

elif page == "Hỗ trợ đa Agent":

    st.markdown(
        '<div class="section-title">Hệ thống hỗ trợ đa Agent</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Nhập câu hỏi và/hoặc dữ liệu sức khỏe. "
        "Coordinator sẽ xác định trạng thái và điều phối luồng xử lý."
    )

    st.markdown("""
    **Các trạng thái xử lý:**

    - S1: Chỉ đánh giá nguy cơ.
    - S2: Chỉ tra cứu tri thức.
    - S3: Kết hợp đánh giá nguy cơ và tra cứu tri thức.
    """)

    with st.form("multi_agent_form"):

        st.subheader("1. Câu hỏi tri thức")

        multi_question = st.text_area(
            "Nhập câu hỏi (có thể bỏ trống)",
            placeholder="Ví dụ: Tiểu đường type 2 có những triệu chứng nào?",
            max_chars=2000,
            height=100
        )

        st.divider()

        st.subheader("2. Dữ liệu sức khỏe")

        enable_health_data = st.checkbox(
            "Tôi muốn gửi thêm dữ liệu sức khỏe để đánh giá"
        )

        if enable_health_data:

            col1, col2 = st.columns(2)

            with col1:

                multi_pregnancies = st.number_input(
                    "Số lần mang thai",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                    key="multi_pregnancies"
                )

                multi_glucose = st.number_input(
                    "Glucose (mg/dL)",
                    min_value=0.0,
                    value=120.0,
                    step=1.0,
                    key="multi_glucose"
                )

                multi_bp = st.number_input(
                    "Blood Pressure (mmHg)",
                    min_value=0.0,
                    value=70.0,
                    step=1.0,
                    key="multi_bp"
                )

                multi_skin = st.number_input(
                    "Skin Thickness (mm)",
                    min_value=0.0,
                    value=20.0,
                    step=1.0,
                    key="multi_skin"
                )

            with col2:

                multi_insulin = st.number_input(
                    "Insulin (mu U/mL)",
                    min_value=0.0,
                    value=80.0,
                    step=1.0,
                    key="multi_insulin"
                )

                multi_bmi = st.number_input(
                    "BMI",
                    min_value=0.0,
                    value=25.0,
                    step=0.1,
                    key="multi_bmi"
                )

                multi_pedigree = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0,
                    value=0.5,
                    step=0.01,
                    key="multi_pedigree"
                )

                multi_age = st.number_input(
                    "Tuổi",
                    min_value=1,
                    max_value=120,
                    value=30,
                    step=1,
                    key="multi_age"
                )

        multi_submitted = st.form_submit_button(
            "Gửi yêu cầu đến hệ đa Agent",
            type="primary",
            use_container_width=True
        )

    if multi_submitted:

        health_data = None

        if enable_health_data:

            health_data = {
                "Pregnancies": multi_pregnancies,
                "Glucose": multi_glucose,
                "BloodPressure": multi_bp,
                "SkinThickness": multi_skin,
                "Insulin": multi_insulin,
                "BMI": multi_bmi,
                "DiabetesPedigreeFunction": multi_pedigree,
                "Age": multi_age
            }

        if not multi_question.strip() and health_data is None:

            st.warning(
                "Vui lòng nhập câu hỏi hoặc chọn gửi dữ liệu sức khỏe."
            )

        else:

            system = get_system()

            if system is not None:

                with st.spinner(
                    "Coordinator đang điều phối các Agent..."
                ):

                    try:
                        # S1, S2 hoặc S3 được Coordinator xác định.
                        result = system.process(
                            question=multi_question.strip() or None,
                            health_data=health_data
                        )

                        show_agent_result(result)

                    except Exception as error:
                        st.error(
                            "Hệ đa Agent gặp lỗi khi xử lý yêu cầu."
                        )
                        st.exception(error)


# =========================================================
# 11. CANH BAO Y KHOA
# =========================================================

st.divider()

st.markdown("""
<div class="disclaimer">
    <strong>Lưu ý y khoa:</strong>
    DiaCare là hệ thống hỗ trợ nghiên cứu và tham khảo.
    Kết quả phân loại từ mô hình MLP không phải chẩn đoán y khoa,
    không thay thế xét nghiệm, chẩn đoán hoặc tư vấn của bác sĩ.
    Điểm số mô hình không đồng nghĩa với xác suất mắc bệnh đã được
    hiệu chuẩn lâm sàng. Các giá trị Integrated Gradients chỉ giải
    thích hành vi của mô hình, không chứng minh nguyên nhân gây bệnh.
</div>
""", unsafe_allow_html=True)
