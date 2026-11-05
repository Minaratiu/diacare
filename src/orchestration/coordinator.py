
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.prediction.prediction_agent import PredictionAgent
from src.orchestration.q_learning import QLearningRouter
from src.rules.rule_engine import RuleEngine
from src.agents.support_agent import SupportAgent


class CoordinatorAgent:
    """Điều phối các Agent trong hệ thống DiaCare."""

    def __init__(self):
        self.router = QLearningRouter()
        self.rules = RuleEngine()
        self.support = SupportAgent()

        self.knowledge_agent = None
        self.prediction_agent = PredictionAgent()
        self.risk_assessment_agent = RiskAssessmentAgent()

    def _get_knowledge_agent(self):
        if self.knowledge_agent is None:
            self.knowledge_agent = KnowledgeAgent()
        return self.knowledge_agent

    def process(
        self,
        question=None,
        health_data=None,
        conversation_history=None,
    ):
        """
        question: câu hỏi y khoa, có thể bỏ trống.
        health_data: dictionary gồm 8 thuộc tính, có thể bỏ trống.
        """

        has_question = (
            isinstance(question, str) and bool(question.strip())
        )
        has_health_data = health_data is not None

        if not has_question and not has_health_data:
            return {
                "success": False,
                "message": "Vui long nhap cau hoi hoac du lieu suc khoe."
            }

        # Xác định State
        if has_question and has_health_data:
            state = "S3"
        elif has_health_data:
            state = "S1"
        else:
            state = "S2"

        # Kiểm tra dữ liệu trước khi điều phối
        if has_health_data:
            valid, message = self.rules.validate_assessment(health_data)

            if not valid:
                return {
                    "success": False,
                    "state": state,
                    "error_type": "invalid_input",
                    "message": message
                }

        if has_question:
            valid, message = self.rules.validate_question(question)

            if not valid:
                return {
                    "success": False,
                    "state": state,
                    "message": message
                }

        # Q-learning lựa chọn luồng xử lý
        action = self.router.choose_action(state)

        results = {
            "success": True,
            "state": state,
            "action": action,
            "prediction": None,
            "risk_assessment": None,
            "knowledge": None
        }

        # Luồng đánh giá sức khỏe
        if action in ["A1", "A3"] and has_health_data:

            # MLP dự đoán và giải thích bằng Integrated Gradients
            results["prediction"] = self.prediction_agent.predict(
                health_data
            )

            # Đánh giá các chỉ số theo ngưỡng tham chiếu y tế
            results["risk_assessment"] = (
                self.risk_assessment_agent.assess(health_data)
            )

        # Luồng tra cứu kiến thức
        if action in ["A2", "A3"] and has_question:
            knowledge_agent = self._get_knowledge_agent()
            if conversation_history is None:
                results["knowledge"] = knowledge_agent.answer(question)
            else:
                results["knowledge"] = knowledge_agent.answer(
                    question,
                    conversation_history=conversation_history,
                )

        # Kiểm tra kết quả điều phối
        if has_health_data and results["prediction"] is None:
            results["success"] = False
            results["message"] = (
                "Q-learning chua chon luong du doan phu hop."
            )
            return results

        if has_question and results["knowledge"] is None:
            results["success"] = False
            results["message"] = (
                "Q-learning chua chon luong tra cuu phu hop."
            )
            return results

        # Tổng hợp kết quả từ các Agent
        results["response"] = self.support.generate(results)

        return results

    def close(self):
        if self.knowledge_agent is not None:
            self.knowledge_agent.close()

        if self.prediction_agent is not None:
            del self.prediction_agent
