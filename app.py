
import streamlit as st
import pandas as pd
import logging


LOGGER = logging.getLogger(__name__)

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
    :root {
        --diacare-blue: #1769e0;
        --diacare-blue-dark: #1154bd;
        --diacare-blue-soft: #edf5ff;
        --diacare-ink: #182b46;
        --diacare-muted: #63748b;
        --diacare-border: #e3eaf3;
        --diacare-surface: #ffffff;
    }

    .stApp {
        background:
            linear-gradient(
                135deg,
                rgba(231, 241, 255, 0.72) 0%,
                rgba(248, 250, 253, 0.96) 48%,
                rgba(235, 243, 253, 0.72) 100%
            ),
            #f6f8fb;
        background-attachment: fixed;
        color: var(--diacare-ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
            BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding: 1.65rem clamp(1.25rem, 3vw, 2.8rem) 2.5rem;
    }

    section[data-testid="stSidebar"] {
        width: 238px !important;
        min-width: 238px !important;
        background: var(--diacare-surface);
        border-right: 1px solid var(--diacare-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem 0.8rem 1.1rem;
    }

    .diacare-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.1rem 0.35rem 0.2rem;
    }

    .diacare-brand-mark {
        display: grid;
        width: 34px;
        height: 34px;
        place-items: center;
        border-radius: 10px;
        color: #fff;
        background: var(--diacare-blue);
        font-size: 18px;
    }

    .diacare-brand-name {
        color: var(--diacare-ink);
        font-size: 17px;
        font-weight: 720;
        letter-spacing: -0.035em;
        line-height: 1.15;
    }

    .diacare-brand-caption {
        margin-top: 3px;
        color: var(--diacare-muted);
        font-size: 9px;
        letter-spacing: 0.06em;
    }

    .main-title {
        color: var(--diacare-ink);
        font-size: 25px;
        font-weight: 720;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        color: var(--diacare-muted);
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 1.45rem;
    }

    h3 {
        font-size: 1.15rem;
    }

    h4 {
        font-size: 1rem;
    }

    .section-title {
        color: var(--diacare-ink);
        font-size: 21px;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }

    .info-box {
        background-color: var(--diacare-blue-soft);
        padding: 0.8rem 1rem;
        border: 1px solid #dce9f8;
        border-radius: 10px;
        margin: 0.7rem 0;
    }

    .disclaimer {
        background-color: #f1f5f9;
        color: #52647a;
        padding: 0.8rem 1rem;
        border: 1px solid var(--diacare-border);
        border-radius: 10px;
        font-size: 13px;
        line-height: 1.55;
    }

    h1, h2, h3, h4 {
        color: var(--diacare-ink);
        letter-spacing: -0.025em;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: #74839a;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.11em;
        margin: 0 0 0.35rem 0.4rem;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.15rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        min-height: 38px;
        padding: 0.3rem 0.55rem;
        border: 1px solid transparent;
        border-radius: 8px;
        color: #52647a;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
        background: #f3f7fc;
        color: var(--diacare-blue-dark);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: var(--diacare-blue-soft);
        border-color: #d7e8ff;
        color: var(--diacare-blue-dark);
        font-weight: 680;
    }

    [data-testid="stSidebar"] input[type="radio"] {
        accent-color: var(--diacare-blue);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
        display: none;
    }

    [data-testid="stButton"] > button,
    [data-testid="stFormSubmitButton"] > button {
        border: 1px solid var(--diacare-border);
        border-radius: 8px;
        min-height: 38px;
        font-weight: 600;
    }

    [data-testid="stButton"] > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: #bfd5ef;
        color: var(--diacare-blue-dark);
        background: var(--diacare-blue-soft);
    }

    [data-testid="stFormSubmitButton"] > button[kind="primary"],
    [data-testid="stButton"] > button[kind="primary"] {
        background: var(--diacare-blue);
        border-color: var(--diacare-blue);
        color: #ffffff;
    }

    [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--diacare-blue-dark);
        border-color: var(--diacare-blue-dark);
        color: #ffffff;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stChatInput"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-radius: 10px;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stChatInput"] textarea:focus {
        border-color: #60a5fa;
        box-shadow: 0 0 0 1px #60a5fa;
    }

    [data-testid="stChatMessage"] {
        background: var(--diacare-surface);
        border: 1px solid #e8edf4;
        border-radius: 11px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.55rem;
    }

    [data-testid="stExpander"] {
        background: var(--diacare-surface);
        border: 1px solid var(--diacare-border);
        border-radius: 9px;
    }

    div[data-testid="stMetric"] {
        background-color: var(--diacare-surface);
        border: 1px solid var(--diacare-border);
        padding: 0.75rem 0.9rem;
        border-radius: 10px;
    }

    div[data-testid="stMetric"] label {
        color: var(--diacare-muted);
    }

    [data-testid="stForm"] {
        padding: 1rem 1.1rem 1.1rem;
        border: 1px solid var(--diacare-border);
        border-radius: 12px;
        background: var(--diacare-surface);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--diacare-border);
        border-radius: 9px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 11px;
    }

    hr {
        border-color: var(--diacare-border);
    }

    .dashboard-intro {
        max-width: 720px;
        margin: 0 0 1.5rem;
        color: var(--diacare-muted);
        font-size: 14px;
        line-height: 1.55;
    }

    .dashboard-card-copy {
        margin-bottom: 0.65rem;
        color: var(--diacare-muted);
        font-size: 13px;
        line-height: 1.45;
    }

    .dashboard-card-title {
        margin: 0.1rem 0 0.55rem;
        color: var(--diacare-ink);
        font-size: 16px;
        font-weight: 680;
        letter-spacing: -0.02em;
        line-height: 1.35;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--diacare-border);
        border-radius: 10px;
        background: var(--diacare-surface);
    }

    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.25rem 0.85rem 2rem;
        }

        [data-testid="stForm"] {
            padding: 0.8rem;
        }

        section[data-testid="stSidebar"] {
            width: min(238px, 82vw) !important;
            min-width: min(238px, 82vw) !important;
        }
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
    return get_coordinator()


def _set_active_page(page_name):
    st.session_state.main_navigation = page_name


# =========================================================
# 4. CAC HAM HIEN THI KET QUA
# =========================================================

def show_explanation(explanation):
    """
    Hiển thị kết quả giải thích Integrated Gradients.
    """

    st.markdown("#### Đóng góp của các yếu tố vào đầu ra mô hình")

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


def show_prediction(prediction, risk_assessment=None, health_data=None):
    if not prediction:
        return

    if not prediction.get("success"):
        st.error("Không thể thực hiện đánh giá.")
        return

    predicted_class = prediction.get("predicted_class")
    score = prediction.get("model_score", 0)
    threshold = prediction.get("threshold", 0.5)

    st.title("KẾT QUẢ ĐÁNH GIÁ SỨC KHỎE")

    if predicted_class == 1:
        st.subheader("Có dấu hiệu nguy cơ")
        overview = (
            "Mô hình AI nhận thấy các chỉ số sức khỏe của bạn thuộc nhóm "
            "có dấu hiệu nguy cơ mắc tiểu đường. Điều này không đồng nghĩa "
            "với việc bạn đã mắc bệnh."
        )
    else:
        st.subheader("Chưa ghi nhận dấu hiệu nguy cơ")
        overview = (
            "Mô hình AI phân loại các chỉ số sức khỏe bạn cung cấp vào "
            "nhóm chưa ghi nhận dấu hiệu nguy cơ theo mô hình. Đây là "
            "kết quả tham khảo, không đồng nghĩa với việc bạn chắc chắn "
            "không mắc bệnh."
        )

    st.caption(
        "Kết quả đánh giá dựa trên các chỉ số sức khỏe bạn cung cấp"
    )

    st.markdown("### 1. Kết quả đánh giá")
    st.write(overview)

    st.markdown("### 2. Các chỉ số sức khỏe")

    assessment_findings = {
        finding.get("name"): finding
        for finding in (risk_assessment or {}).get("findings", [])
    }

    def format_value(value):
        if value is None:
            return "Chưa cung cấp"

        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)

    glucose_finding = assessment_findings.get("Glucose", {})
    glucose_status = glucose_finding.get("status", "")
    if glucose_status == "Chưa đủ thông tin":
        glucose_comment = (
            "Cần xác định loại xét nghiệm (ví dụ lúc đói hoặc OGTT) "
            "để diễn giải phù hợp."
        )
    elif glucose_status:
        glucose_comment = (
            f"{glucose_status}. "
            f"{glucose_finding.get('interpretation', '')}"
        ).strip()
    else:
        glucose_comment = "Chưa có đánh giá tham chiếu."

    bmi_finding = assessment_findings.get("BMI", {})
    bmi_comment = bmi_finding.get(
        "status",
        "Chưa có đánh giá tham chiếu."
    )
    if bmi_finding.get("interpretation"):
        bmi_comment += f". {bmi_finding['interpretation']}"

    health_data = health_data or {}
    health_rows = []
    feature_rows = [
        ("Pregnancies", "Số lần mang thai", "lần"),
        ("Glucose", "Glucose", "mg/dL"),
        ("BloodPressure", "Huyết áp tâm trương", "mmHg"),
        ("SkinThickness", "Độ dày da", "mm"),
        ("Insulin", "Insulin", "mu U/mL"),
        ("BMI", "BMI", "kg/m²"),
        ("DiabetesPedigreeFunction", "Yếu tố di truyền", ""),
        ("Age", "Tuổi", "tuổi"),
    ]

    feature_comments = {
        "Pregnancies": (
            "Được mô hình sử dụng; chưa đối chiếu ngưỡng tham khảo riêng."
        ),
        "Glucose": glucose_comment,
        "BloodPressure": (
            "Chỉ có huyết áp tâm trương; chưa đủ thông tin để đánh giá "
            "huyết áp đầy đủ."
        ),
        "SkinThickness": (
            "Được mô hình sử dụng; chưa đối chiếu ngưỡng tham khảo riêng."
        ),
        "Insulin": (
            "Được mô hình sử dụng; chưa đối chiếu ngưỡng tham khảo riêng."
        ),
        "BMI": bmi_comment,
        "DiabetesPedigreeFunction": (
            "Đặc trưng của mô hình; không phải kết quả xét nghiệm."
        ),
        "Age": "Cần xem xét cùng các yếu tố sức khỏe khác.",
    }

    for feature, label, unit in feature_rows:
        value = format_value(health_data.get(feature))
        health_rows.append({
            "Chỉ số": label,
            "Giá trị": f"{value} {unit}".strip(),
            "Nhận xét": feature_comments[feature],
        })

    st.dataframe(
        pd.DataFrame(health_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 3. Những yếu tố ảnh hưởng đến kết quả")
    st.write(
        "Biểu đồ thể hiện các chỉ số đóng góp vào kết quả đánh giá của "
        "mô hình AI. Yếu tố đóng góp làm tăng hoặc giảm đầu ra mô hình, "
        "không đồng nghĩa với nguyên nhân gây bệnh hay mức độ nguy hiểm "
        "của chỉ số đó."
    )
    st.caption(
        f"Điểm đầu ra mô hình: {format_value(score)} · "
        f"Ngưỡng phân loại: {format_value(threshold)}. "
        "Điểm này không phải xác suất mắc bệnh đã được hiệu chuẩn lâm sàng."
    )

    show_explanation(prediction.get("explanation"))

    st.markdown("### 4. Bạn nên làm gì tiếp theo?")
    recommendations = [
        "Trao đổi với nhân viên y tế để được đánh giá tình trạng sức khỏe.",
        "Hỏi về việc thực hiện xét nghiệm HbA1c hoặc glucose huyết tương "
        "lúc đói.",
        "Duy trì chế độ ăn uống cân đối, vận động phù hợp và theo dõi "
        "sức khỏe định kỳ.",
    ]

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    st.markdown("### Lưu ý")
    st.warning(
        "Đây là kết quả sàng lọc tham khảo từ mô hình AI, không phải "
        "chẩn đoán y khoa và không thay thế việc thăm khám, xét nghiệm "
        "hoặc tư vấn của nhân viên y tế. Không tự ý sử dụng thuốc hoặc "
        "thay đổi phương pháp điều trị dựa trên kết quả này."
    )

    sources = (risk_assessment or {}).get("sources", [])
    if sources:
        with st.expander("Nguồn tham khảo"):
            for source in sources:
                name = source.get("name", "Nguồn y tế")
                url = source.get("url")
                if url:
                    st.markdown(f"- [{name}]({url})")
                else:
                    st.write(f"- {name}")


def show_assessment_result(result):
    """Render the concise, user-facing outcome of a health assessment."""

    st.title("KẾT QUẢ ĐÁNH GIÁ SỨC KHỎE")

    prediction = result.get("prediction") if result else None
    if result and result.get("success") and prediction:
        if prediction.get("success") and prediction.get("predicted_class") == 0:
            st.subheader("Chưa ghi nhận dấu hiệu nguy cơ")
            st.write(
                "Các chỉ số bạn cung cấp hiện chưa ghi nhận dấu hiệu nguy cơ "
                "theo đánh giá của hệ thống. Đây là tín hiệu tích cực. Bạn hãy "
                "tiếp tục duy trì chế độ ăn uống lành mạnh, vận động thường "
                "xuyên và theo dõi sức khỏe định kỳ."
            )
        elif prediction.get("success") and prediction.get("predicted_class") == 1:
            st.subheader("Có dấu hiệu nguy cơ")
            st.write(
                "Một số đặc điểm trong các chỉ số sức khỏe của bạn được hệ "
                "thống xếp vào nhóm có dấu hiệu nguy cơ mắc tiểu đường. Bạn "
                "không cần quá lo lắng, nhưng nên chủ động trao đổi với nhân "
                "viên y tế và cân nhắc kiểm tra đường huyết để được đánh giá "
                "chính xác hơn."
            )
        else:
            st.subheader("Chưa có kết quả đánh giá")
            st.write(
                "Hệ thống hiện chưa thể thực hiện đánh giá do xảy ra lỗi "
                "trong quá trình xử lý. Vui lòng thử lại sau. Thông tin sức "
                "khỏe bạn đã nhập không nên được xem là kết quả chẩn đoán."
            )
    elif result and result.get("error_type") == "invalid_input":
        st.subheader("Chưa thể đánh giá sức khỏe")
        st.write(
            "Thông tin bạn cung cấp chưa đầy đủ hoặc có một số giá trị chưa "
            "phù hợp. Vui lòng kiểm tra và nhập lại các chỉ số để hệ thống "
            "có thể thực hiện đánh giá."
        )
    else:
        st.subheader("Chưa có kết quả đánh giá")
        st.write(
            "Hệ thống hiện chưa thể thực hiện đánh giá do xảy ra lỗi trong "
            "quá trình xử lý. Vui lòng thử lại sau. Thông tin sức khỏe bạn "
            "đã nhập không nên được xem là kết quả chẩn đoán."
        )

def show_knowledge(knowledge, compact=False):
    if not knowledge:
        return

    if not compact:
        st.subheader("Kết quả tra cứu tri thức")

    if not knowledge.get("success"):
        st.error(
            knowledge.get(
                "message",
                "Không thể truy xuất tri thức."
            )
        )
        return

    answer_mode = knowledge.get("answer_mode")
    status_message = knowledge.get("message")
    if status_message and answer_mode in (
        "evidence_only",
        "insufficient_evidence",
    ):
        st.warning(status_message)

    answer = knowledge.get("answer")
    if answer:
        if not compact:
            st.markdown("#### Câu trả lời")
        st.markdown(answer)

    sources = knowledge.get("sources", [])
    if not sources:
        sources = knowledge.get("results", [])

    if not sources:
        return

    st.markdown("#### NGUỒN THAM KHẢO")
    for index, source in enumerate(sources, start=1):
        title = (
            source.get("title")
            or source.get("name")
            or source.get("source_name")
            or f"Tài liệu {index}"
        )
        organization = source.get("organization") or "Không xác định"
        url = source.get("url") or source.get("source_url")
        citation_ids = source.get("citation_ids") or [index]
        citation_label = ", ".join(
            str(citation_id) for citation_id in citation_ids
        )

        with st.expander(f"[{citation_label}] {title}"):
            st.write(f"**Tổ chức:** {organization}")
            if url:
                st.markdown(f"**Bài viết gốc:** [{url}]({url})")
            else:
                st.write("**Bài viết gốc:** Không có đường dẫn xác minh.")


def show_agent_result(result, health_data=None, show_routing_info=True):
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

    if show_routing_info:
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
        show_prediction(
            result["prediction"],
            risk_assessment=result.get("risk_assessment"),
            health_data=health_data,
        )

    if result.get("knowledge") is not None:
        show_knowledge(result["knowledge"])

    if result.get("response"):
        with st.expander("Nội dung tổng hợp từ Support Agent"):
            st.text(result["response"])


# =========================================================
# 5. THANH DIEU HUONG
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="diacare-brand">
            <div class="diacare-brand-mark">✚</div>
            <div>
                <div class="diacare-brand-name">DiaCare</div>
                <div class="diacare-brand-caption">HEALTHCARE ASSISTANT</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page_icons = {
        "Tổng quan": "⌂  Tổng quan",
        "Đánh giá nguy cơ": "◉  Đánh giá nguy cơ",
        "Tra cứu tri thức": "⌕  Tra cứu tri thức",
        "Hỗ trợ đa Agent": "✳  Hỗ trợ đa Agent",
    }

    page = st.radio(
        "CHỨC NĂNG",
        [
            "Tổng quan",
            "Đánh giá nguy cơ",
            "Tra cứu tri thức",
            "Hỗ trợ đa Agent"
        ],
        index=0,
        format_func=page_icons.get,
        key="main_navigation",
    )

    st.divider()

    with st.expander("Công nghệ sử dụng"):
        st.caption("Deep Learning · MLP")
        st.caption("Neo4j Knowledge Graph")
        st.caption("Retrieval-Augmented Generation")
        st.caption("Q-Learning · Rule Engine")

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
# 7. CHUC NANG DANH GIA NGUY CO
# =========================================================

if page == "Tổng quan":

    st.markdown(
        '<div class="section-title">Tổng quan</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dashboard-intro">'
        "Truy cập nhanh các công cụ chăm sóc sức khỏe và tra cứu tiểu đường."
        "</div>",
        unsafe_allow_html=True,
    )

    dashboard_cards = [
        (
            "Đánh giá nguy cơ",
            "◉",
            "Tham khảo kết quả phân loại dựa trên các chỉ số sức khỏe.",
        ),
        (
            "Tra cứu tri thức",
            "⌕",
            "Tìm câu trả lời về tiểu đường cùng nguồn y tế tham khảo.",
        ),
        (
            "Hỗ trợ đa Agent",
            "✳",
            "Gửi câu hỏi hoặc dữ liệu sức khỏe để được hỗ trợ.",
        ),
    ]

    st.markdown("#### Công cụ")
    card_columns = st.columns(3)
    for column, (target_page, icon, description) in zip(
        card_columns,
        dashboard_cards,
    ):
        with column:
            with st.container(border=True):
                st.markdown(
                    f'<div class="dashboard-card-title">'
                    f"{icon}&nbsp; {target_page}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="dashboard-card-copy">{description}</div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    f"Mở {target_page}",
                    key=f"dashboard_open_{target_page}",
                    use_container_width=True,
                    on_click=_set_active_page,
                    args=(target_page,),
                )

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

        status = None

        try:
            with st.status(
                "Đang cập nhật kết quả đánh giá...",
                expanded=True
            ) as status:
                st.write(
                    "Đang kiểm tra các chỉ số và chạy mô hình MLP."
                )

                system = get_system()

                if system is None:
                    status.update(
                        label="Không thể khởi tạo hệ thống đánh giá.",
                        state="error",
                        expanded=True
                    )
                else:
                    # Coordinator thực hiện Rule Engine,
                    # Q-Learning Router và Prediction Agent.
                    result = system.process(
                        health_data=health_data
                    )

                    status.update(
                        label="Đã cập nhật kết quả đánh giá.",
                        state="complete",
                        expanded=False
                    )

            if system is not None:
                show_assessment_result(result)
            else:
                show_assessment_result(None)

        except Exception:
            LOGGER.exception("Health assessment failed")
            if status is not None:
                status.update(
                    label="Đánh giá không hoàn tất.",
                    state="error",
                    expanded=True
                )

            show_assessment_result(None)


# =========================================================
# 9. CHUC NANG TRA CUU TRI THUC
# =========================================================

elif page == "Tra cứu tri thức":

    st.markdown(
        '<div class="section-title">Chatbot hỗ trợ bệnh tiểu đường</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Đặt câu hỏi về bệnh tiểu đường, yếu tố nguy cơ, xét nghiệm, dinh "
        "dưỡng hoặc chăm sóc sức khỏe. DiaCare trả lời dựa trên tài liệu "
        "Neo4j và các website y tế chính thức khi truy xuất được."
    )

    if "knowledge_chat_history" not in st.session_state:
        st.session_state.knowledge_chat_history = []

    clear_column, _ = st.columns([1, 5])
    with clear_column:
        if st.button("Xóa cuộc trò chuyện", key="clear_knowledge_chat"):
            st.session_state.knowledge_chat_history = []
            st.rerun()

    if not st.session_state.knowledge_chat_history:
        with st.chat_message("assistant"):
            st.markdown(
                "Xin chào! Tôi có thể giúp bạn tìm hiểu thông tin về bệnh "
                "tiểu đường dựa trên tài liệu y tế có nguồn gốc rõ ràng. "
                "Bạn muốn bắt đầu từ chủ đề nào?"
            )

    for turn in st.session_state.knowledge_chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn.get("knowledge"):
                show_knowledge(turn["knowledge"], compact=True)
            else:
                st.error(
                    turn.get(
                        "error",
                        "DiaCare chưa thể xử lý câu hỏi này. Vui lòng thử lại.",
                    )
                )

    chat_input = st.chat_input(
        "Nhập câu hỏi về bệnh tiểu đường...",
        max_chars=2000,
    )
    question = chat_input

    if question:
        previous_turns = st.session_state.knowledge_chat_history[-3:]
        conversation_history = []
        for turn in previous_turns:
            conversation_history.append({
                "role": "user",
                "content": turn["question"],
            })
            knowledge = turn.get("knowledge") or {}
            answer = knowledge.get("answer")
            if answer:
                conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                })

        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                system = get_system()
                with st.spinner(
                    "DiaCare đang tìm tài liệu y tế và chuẩn bị câu trả lời..."
                ):
                    result = system.process(
                        question=question.strip(),
                        conversation_history=conversation_history,
                    )

                if result.get("success") and result.get("knowledge"):
                    knowledge = result["knowledge"]
                    show_knowledge(knowledge, compact=True)
                    st.session_state.knowledge_chat_history.append({
                        "question": question.strip(),
                        "knowledge": knowledge,
                    })
                else:
                    error_message = result.get(
                        "message",
                        "DiaCare chưa tìm được câu trả lời. Vui lòng thử lại.",
                    )
                    st.error(error_message)
                    st.session_state.knowledge_chat_history.append({
                        "question": question.strip(),
                        "error": error_message,
                    })
            except Exception:
                LOGGER.exception("Medical knowledge chatbot request failed")
                error_message = (
                    "Không thể thực hiện tra cứu lúc này. Vui lòng thử lại sau."
                )
                st.error(error_message)
                st.session_state.knowledge_chat_history.append({
                    "question": question.strip(),
                    "error": error_message,
                })


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

                        show_agent_result(
                            result,
                            health_data=health_data,
                        )

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
