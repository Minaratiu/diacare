
import os
import numpy as np

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    """
    Truy xuất kiến thức y khoa từ Knowledge Graph Neo4j
    bằng Sentence Transformers và Cosine Similarity.
    """

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
            s.url AS source_url
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
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Vì vector đã chuẩn hóa, tích vô hướng tương đương cosine similarity
        scores = np.dot(
            self.embeddings,
            question_embedding
        )

        # Lấy các vị trí có điểm tương đồng cao nhất
        top_k = max(1, min(top_k, len(self.documents)))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            document = self.documents[index]

            results.append({
                "chunk_id": document["chunk_id"],
                "content": document["content"],
                "source_name": document["source_name"],
                "organization": document["organization"],
                "source_url": document["source_url"],
                "score": float(scores[index])
            })

        return results

    def close(self):
        """Đóng kết nối Neo4j."""
        self.driver.close()
