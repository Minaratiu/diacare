
import os
import re
import unicodedata
from collections import Counter
from math import log

import numpy as np

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    """
    Truy xuất kiến thức y khoa từ Knowledge Graph Neo4j
    bằng Sentence Transformers và Cosine Similarity.
    """

    INTENT_CUES = {
        "comparison": (
            "so sánh",
            "khác nhau",
            "phân biệt",
            "versus",
            "vs",
            "compare",
            "difference between",
        ),
        "definition": (
            "là gì",
            "nghĩa là gì",
            "định nghĩa",
            "what is",
            "define",
            "meaning of",
        ),
        "cause": (
            "nguyên nhân",
            "do đâu",
            "vì sao",
            "tại sao",
            "cause of",
            "causes of",
            "why does",
        ),
        "symptoms": (
            "triệu chứng",
            "dấu hiệu",
            "biểu hiện",
            "symptom",
            "signs of",
        ),
        "risk_factors": (
            "yếu tố nguy cơ",
            "nguy cơ",
            "risk factor",
            "risk of",
        ),
        "diagnosis": (
            "chẩn đoán",
            "xét nghiệm",
            "sàng lọc",
            "diagnos",
            "screening",
            "test for",
            "hba1c",
            "a1c",
        ),
        "treatment": (
            "điều trị",
            "chữa bệnh",
            "thuốc",
            "kiểm soát",
            "treatment",
            "medication",
            "manage diabetes",
        ),
    }
    INTENT_EXPANSIONS = {
        "comparison": "comparison differences compared with",
        "definition": "definition overview meaning",
        "cause": "causes etiology why",
        "symptoms": "symptoms signs",
        "risk_factors": "risk factors",
        "diagnosis": "diagnosis testing screening",
        "treatment": "treatment management medication",
    }
    INTENT_CONTENT_TERMS = {
        "comparison": (
            "comparison",
            "difference",
            "differences",
            "differ",
            "versus",
            "so sánh",
            "khác nhau",
            "phân biệt",
        ),
        "definition": (
            "definition",
            "defined as",
            "overview",
            "định nghĩa",
            "khái niệm",
            "là bệnh",
        ),
        "cause": (
            "cause",
            "causes",
            "caused by",
            "etiology",
            "nguyên nhân",
            "gây ra",
            "xảy ra khi",
            "occurs when",
        ),
        "symptoms": (
            "symptom",
            "symptoms",
            "sign",
            "signs",
            "manifestation",
            "triệu chứng",
            "dấu hiệu",
            "biểu hiện",
        ),
        "risk_factors": (
            "risk factor",
            "risk factors",
            "increased risk",
            "yếu tố nguy cơ",
            "nguy cơ",
        ),
        "diagnosis": (
            "diagnosis",
            "diagnosed",
            "screening",
            "blood test",
            "xét nghiệm",
            "chẩn đoán",
            "sàng lọc",
        ),
        "treatment": (
            "treatment",
            "treatments",
            "management",
            "medication",
            "insulin therapy",
            "insulin treatment",
            "insulin injections",
            "therapy",
            "điều trị",
            "sử dụng thuốc điều trị",
            "thuốc điều trị",
            "thuốc tiểu đường",
            "thuốc hạ đường huyết",
            "kiểm soát",
        ),
    }
    COMPARISON_CONCEPTS = {
        "type_1_diabetes": (
            "type 1 diabetes",
            "diabetes type 1",
            "tiểu đường type 1",
            "tiểu đường tuýp 1",
            "đái tháo đường type 1",
            "đái tháo đường tuýp 1",
            "loại 1 diabetes",
            "type 1",
            "tuýp 1",
        ),
        "type_2_diabetes": (
            "type 2 diabetes",
            "diabetes type 2",
            "tiểu đường type 2",
            "tiểu đường tuýp 2",
            "đái tháo đường type 2",
            "đái tháo đường tuýp 2",
            "loại 2 diabetes",
            "type 2",
            "tuýp 2",
        ),
        "gestational_diabetes": (
            "gestational diabetes",
            "tiểu đường thai kỳ",
            "đái tháo đường thai kỳ",
            "gestational",
        ),
        "prediabetes": (
            "prediabetes",
            "pre diabetes",
            "tiền tiểu đường",
            "tiền đái tháo đường",
        ),
        "hypoglycemia": (
            "hypoglycemia",
            "hypoglycaemia",
            "hạ đường huyết",
            "đường huyết thấp",
            "low blood sugar",
        ),
        "hyperglycemia": (
            "hyperglycemia",
            "hyperglycaemia",
            "tăng đường huyết",
            "đường huyết cao",
            "high blood sugar",
        ),
    }
    QUERY_STOP_WORDS = {
        "a", "an", "and", "are", "about", "be", "between", "can", "common",
        "cua", "co", "cach", "cho", "gi", "hay", "how", "is", "la", "nao",
        "of", "or", "the", "the nao", "thi", "to", "ve", "what", "which", "with",
        "va", "tai", "vi", "khac", "nhau", "so", "sanh", "phan", "biet",
        "tieu", "duong", "dai", "thao", "benh", "diabetes", "diabetic",
        "blood", "sugar", "glucose",
    }

    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        load_dotenv()

        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE")

        if not all([
            self.uri,
            self.username,
            self.password,
            self.database
        ]):
            raise ValueError("Thieu cau hinh Neo4j trong file .env")

        # Tải mô hình Embedding
        self.model = SentenceTransformer(model_name)

        # Khởi tạo Neo4j Driver
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )

        self.documents = []
        self.embeddings = None

    def load_documents(self):
        """
        Lấy KnowledgeChunk và Source từ Neo4j.
        """

        query = """
        MATCH (k:KnowledgeChunk)-[:CITED_FROM]->(s:Source)
        RETURN
            k.id AS chunk_id,
            k.content AS content,
            s.name AS source_name,
            s.organization AS organization,
            s.url AS source_url,
            properties(s)['published_date'] AS published_date,
            properties(s)['updated_at'] AS updated_at
        ORDER BY k.id
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            self.documents = [dict(record) for record in result]

        if not self.documents:
            raise ValueError("Khong tim thay KnowledgeChunk trong Neo4j")

        # Kiểm tra nội dung và nguồn trước khi tạo Embedding
        for document in self.documents:
            if not document.get("content"):
                raise ValueError(
                    f"KnowledgeChunk thieu content: {document.get('chunk_id')}"
                )

        return len(self.documents)

    def create_embeddings(self):
        """
        Chuyển nội dung KnowledgeChunk thành Vector Embedding.
        """

        if not self.documents:
            self.load_documents()

        contents = [
            document["content"]
            for document in self.documents
        ]

        self.embeddings = self.model.encode(
            contents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return self.embeddings.shape

    @staticmethod
    def _expand_query(question):
        vietnamese_terms = {
            "đường huyết lúc đói": "fasting blood glucose",
            "tăng đường huyết": "high blood glucose hyperglycemia",
            "đái tháo đường": "diabetes",
            "tiểu đường": "diabetes",
            "đường huyết": "blood glucose blood sugar",
            "tiền tiểu đường": "prediabetes",
            "dấu hiệu": "signs symptoms",
            "triệu chứng": "symptoms",
            "nguy cơ": "risk factors",
            "phòng ngừa": "prevention",
            "ngăn ngừa": "prevention",
            "xét nghiệm": "test screening",
            "hba1c": "A1C test",
            "chế độ ăn": "diet nutrition food",
            "dinh dưỡng": "diet nutrition food",
            "ăn uống": "diet nutrition food",
            "biến chứng": "complications",
            "điều trị": "treatment management",
            "kiểm soát": "control management",
            "loại 2": "type 2",
            "type 2": "type 2",
            "thai kỳ": "gestational diabetes pregnancy",
            "mang thai": "gestational diabetes pregnancy",
            "insulin": "insulin therapy blood sugar",
            "hạ đường huyết": "hypoglycemia low blood sugar",
            "đường huyết thấp": "hypoglycemia low blood sugar",
            "bàn chân": "diabetic foot care",
            "thận": "diabetic kidney disease nephropathy",
            "mắt": "diabetic retinopathy eye disease",
            "tim mạch": "cardiovascular disease",
            "cấp cứu": "diabetic ketoacidosis emergency",
            "nhiễm toan": "diabetic ketoacidosis",
            "tê bì": "diabetic neuropathy",
            "biến chứng": "complications",
            "giảm cân": "weight loss diabetes management",
            "vận động": "physical activity exercise diabetes",
            "thuốc": "diabetes medication treatment",
        }
        def normalize(text):
            decomposed = unicodedata.normalize("NFD", text.casefold())
            without_marks = "".join(
                character
                for character in decomposed
                if unicodedata.category(character) != "Mn"
            )
            return without_marks.replace("đ", "d")

        normalized_question = normalize(question)
        matched_terms = [
            (normalize(term), expansion)
            for term, expansion in vietnamese_terms.items()
            if normalize(term) in normalized_question
        ]
        expansions = []
        for term, expansion in matched_terms:
            if any(
                term != other and term in other
                for other, _ in matched_terms
            ):
                continue
            expansion_terms = set(expansion.casefold().split())
            question_terms = set(normalized_question.split())
            if expansion_terms.isdisjoint(question_terms) and expansion not in expansions:
                expansions.append(expansion)

        if not expansions:
            expanded_question = question
        else:
            expanded_question = f"{question} {' '.join(expansions)}"

        intent = RAGPipeline._detect_intent(question)
        intent_expansion = RAGPipeline.INTENT_EXPANSIONS.get(intent)
        if intent_expansion:
            expansion_terms = set(RAGPipeline._tokens(intent_expansion))
            query_terms = set(RAGPipeline._tokens(expanded_question))
            if expansion_terms.isdisjoint(query_terms):
                expanded_question = f"{expanded_question} {intent_expansion}"
        return expanded_question

    @staticmethod
    def _normalize_text(text):
        decomposed = unicodedata.normalize("NFD", str(text).casefold())
        without_marks = "".join(
            character
            for character in decomposed
            if unicodedata.category(character) != "Mn"
        )
        return without_marks.replace("đ", "d")

    @staticmethod
    def _detect_intent(question):
        normalized_question = RAGPipeline._normalize_text(question)
        intent_priority = (
            "comparison",
            "cause",
            "symptoms",
            "risk_factors",
            "diagnosis",
            "treatment",
            "definition",
        )
        for intent in intent_priority:
            cues = RAGPipeline.INTENT_CUES[intent]
            if any(
                (
                    re.search(
                        rf"\b{re.escape(RAGPipeline._normalize_text(cue))}\b",
                        normalized_question,
                    )
                    is not None
                    if len(cue) <= 2
                    else RAGPipeline._normalize_text(cue) in normalized_question
                )
                for cue in cues
            ):
                return intent
        return "general"

    @staticmethod
    def _comparison_concepts(question):
        return [
            concept
            for concept, aliases in RAGPipeline.COMPARISON_CONCEPTS.items()
            if any(
                RAGPipeline._contains_phrase(question, alias)
                for alias in aliases
            )
        ]

    @staticmethod
    def _tokens(text):
        return re.findall(r"[a-z0-9]+", RAGPipeline._normalize_text(text))

    @staticmethod
    def _contains_phrase(text, phrase):
        normalized_text = RAGPipeline._normalize_text(text)
        normalized_phrase = RAGPipeline._normalize_text(phrase)
        phrase_pattern = r"\s+".join(
            re.escape(token)
            for token in normalized_phrase.split()
        )
        return re.search(
            rf"(?<![a-z0-9]){phrase_pattern}(?![a-z0-9])",
            normalized_text,
        ) is not None

    @staticmethod
    def _rank_documents(question, documents, similarities):
        intent = RAGPipeline._detect_intent(question)
        comparison_concepts = RAGPipeline._comparison_concepts(question)
        question_terms = {
            term for term in RAGPipeline._tokens(question)
            if term not in RAGPipeline.QUERY_STOP_WORDS
        }
        document_terms = [
            set(RAGPipeline._tokens(document["content"]))
            for document in documents
        ]
        document_frequency = Counter(
            term
            for terms in document_terms
            for term in question_terms.intersection(terms)
        )

        ranked = []
        for index, (document, terms) in enumerate(
            zip(documents, document_terms)
        ):
            matched_terms = question_terms.intersection(terms)
            if question_terms:
                weights = {
                    term: log(
                        (len(documents) + 1)
                        / (document_frequency.get(term, 0) + 1)
                    ) + 1
                    for term in question_terms
                }
                lexical_score = (
                    sum(weights[term] for term in matched_terms)
                    / sum(weights.values())
                )
            else:
                lexical_score = 0.0

            normalized_content = RAGPipeline._normalize_text(
                document["content"]
            )
            intent_cues = RAGPipeline.INTENT_CONTENT_TERMS.get(intent, ())
            intent_score = float(any(
                RAGPipeline._contains_phrase(normalized_content, cue)
                for cue in intent_cues
            ))

            matched_concepts = [
                concept
                for concept in comparison_concepts
                if any(
                    RAGPipeline._contains_phrase(normalized_content, alias)
                    for alias in RAGPipeline.COMPARISON_CONCEPTS[concept]
                )
            ]
            if (
                intent == "comparison"
                and len(comparison_concepts) >= 2
                and len(matched_concepts) == len(comparison_concepts)
            ):
                intent_score = 1.0
            comparison_score = (
                len(matched_concepts) / len(comparison_concepts)
                if comparison_concepts
                else 0.0
            )

            similarity = float(similarities[index])
            normalized_similarity = max(
                0.0, min(1.0, (similarity + 1.0) / 2.0)
            )
            if intent == "comparison" and len(comparison_concepts) >= 2:
                relevance_score = (
                    0.48 * normalized_similarity
                    + 0.20 * lexical_score
                    + 0.12 * intent_score
                    + 0.20 * comparison_score
                )
            else:
                relevance_score = (
                    0.62 * normalized_similarity
                    + 0.25 * lexical_score
                    + 0.13 * intent_score
                )

            ranked_document = dict(document)
            ranked_document["score"] = float(relevance_score)
            ranked_document["similarity_score"] = similarity
            ranked_document["intent"] = intent
            ranked_document["intent_match"] = bool(intent_score)
            ranked_document["matched_query_terms"] = sorted(matched_terms)
            ranked_document["comparison_concepts_covered"] = matched_concepts
            ranked.append(ranked_document)

        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    def search(self, question, top_k=3):
        """
        Tìm các KnowledgeChunk gần nghĩa nhất với câu hỏi.
        """

        if not question or not question.strip():
            raise ValueError("Cau hoi khong duoc de trong")

        if self.embeddings is None:
            self.create_embeddings()

        # Chuyển câu hỏi thành Vector Embedding
        question_embedding = self.model.encode(
            self._expand_query(question),
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Vì vector đã chuẩn hóa, tích vô hướng tương đương cosine similarity
        scores = np.dot(
            self.embeddings,
            question_embedding
        )

        top_k = max(1, min(int(top_k), len(self.documents)))
        ranked_documents = self._rank_documents(
            question,
            self.documents,
            scores,
        )
        intent = self._detect_intent(question)
        comparison_concepts = self._comparison_concepts(question)
        if intent != "general":
            ranked_documents = [
                document
                for document in ranked_documents
                if document["intent_match"]
            ]
        if intent == "comparison" and len(comparison_concepts) >= 2:
            required_concepts = set(comparison_concepts)
            ranked_documents = [
                document
                for document in ranked_documents
                if required_concepts.issubset(
                    document["comparison_concepts_covered"]
                )
            ]

        return [
            {
                "chunk_id": document["chunk_id"],
                "content": document["content"],
                "source_name": document["source_name"],
                "organization": document["organization"],
                "source_url": document["source_url"],
                "published_date": document.get("published_date"),
                "updated_at": document.get("updated_at"),
                "score": document["score"],
                "similarity_score": document["similarity_score"],
                "intent": document["intent"],
                "intent_match": document["intent_match"],
                "matched_query_terms": document["matched_query_terms"],
                "comparison_concepts_covered": (
                    document["comparison_concepts_covered"]
                ),
            }
            for document in ranked_documents[:top_k]
        ]

    def close(self):
        """Đóng kết nối Neo4j."""
        self.driver.close()
