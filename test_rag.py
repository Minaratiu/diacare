
from src.knowledge.rag_pipeline import RAGPipeline


def main():
    rag = None

    try:
        print("DANG KHOI TAO RAG...")

        rag = RAGPipeline()

        # Nạp kiến thức và nguồn từ Neo4j
        total = rag.load_documents()
        print(f"So KnowledgeChunk: {total}")

        # Tạo Vector Embedding
        dimension = rag.create_embeddings()
        print(f"Kich thuoc Embedding: {dimension}")

        # Thử tìm kiếm bằng tiếng Việt
        question = "Những dấu hiệu thường gặp của bệnh tiểu đường là gì?"

        print(f"\nCau hoi: {question}")
        print("\nKET QUA TIM KIEM:")

        results = rag.search(question, top_k=3)

        for index, item in enumerate(results, start=1):
            print(f"\n--- Ket qua {index} ---")
            print("Chunk ID:", item["chunk_id"])
            print("Noi dung:", item["content"])
            print("Nguon:", item["source_name"])
            print("To chuc:", item["organization"])
            print("Do tuong dong:", round(item["score"], 4))
            print("URL:", item["source_url"])

        print("\nKIEM THU RAG HOAN TAT!")

    except Exception as error:
        print("\nRAG TEST THAT BAI!")
        print("Chi tiet:", error)

    finally:
        if rag is not None:
            rag.close()


if __name__ == "__main__":
    main()
