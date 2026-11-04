
from src.agents.knowledge_agent import KnowledgeAgent


def main():
    agent = None

    try:
        agent = KnowledgeAgent()

        question = "Bệnh tiểu đường có những triệu chứng nào?"

        print("CAU HOI:", question)

        response = agent.answer(question)

        print("\nTRANG THAI:", response["success"])
        print("THONG BAO:", response["message"])

        print("\nKET QUA TRUY XUAT:")

        for item in response["results"]:
            print("\nNoi dung:", item["content"])
            print("Nguon:", item["source_name"])
            print("Do tuong dong:", round(item["score"], 4))

        print("\nNGUON THAM KHAO:")

        for source in response["sources"]:
            print("-", source["name"])
            print("  URL:", source["url"])

        if response["success"]:
            print("\nKNOWLEDGE AGENT HOAT DONG!")
        else:
            print("\nKNOWLEDGE AGENT CO LOI!")

    except Exception as error:
        print("KIEM THU THAT BAI:", error)

    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    main()
