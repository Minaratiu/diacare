
from src.knowledge.rag_pipeline import RAGPipeline


class KnowledgeAgent:
    """
    Agent chịu trách nhiệm tra cứu kiến thức y khoa
    thông qua RAG Pipeline.
    """

    def __init__(self):
        self.rag = RAGPipeline()

    def answer(self, question, top_k=3):
        """
        Tiếp nhận câu hỏi và truy xuất kiến thức liên quan.

        Lưu ý: Hàm này trả về bằng chứng truy xuất,
        chưa tự sinh câu trả lời y khoa bằng LLM.
        """

        if not isinstance(question, str) or not question.strip():
            return {
                "success": False,
                "message": "Vui lòng nhập câu hỏi.",
                "question": question,
                "results": [],
                "sources": []
            }

        try:
            results = self.rag.search(question, top_k=top_k)

            # Không sử dụng kết quả có độ tương đồng quá thấp.
            # Đây là ngưỡng kỹ thuật ban đầu, cần đánh giá bằng test.
            relevant_results = [
                item for item in results
                if item["score"] >= 0.30
            ]

            if not relevant_results:
                return {
                    "success": True,
                    "message": (
                        "Chưa tìm thấy tài liệu đủ liên quan "
                        "để trả lời câu hỏi."
                    ),
                    "question": question,
                    "results": [],
                    "sources": []
                }

            # Loại bỏ nguồn trùng lặp
            sources = []
            seen_urls = set()

            for item in relevant_results:
                url = item.get("source_url")

                if url and url not in seen_urls:
                    seen_urls.add(url)

                    sources.append({
                        "name": item.get("source_name"),
                        "organization": item.get("organization"),
                        "url": url
                    })

            return {
                "success": True,
                "message": "Truy xuất kiến thức thành công.",
                "question": question,
                "results": relevant_results,
                "sources": sources
            }

        except Exception as error:
            return {
                "success": False,
                "message": f"Lỗi truy xuất kiến thức: {error}",
                "question": question,
                "results": [],
                "sources": []
            }

    def close(self):
        self.rag.close()
