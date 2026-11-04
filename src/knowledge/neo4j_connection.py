
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


# Đọc thông tin kết nối từ file .env ở thư mục gốc dự án
load_dotenv()


def get_neo4j_driver():
    """Tạo kết nối đến Neo4j AuraDB."""

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not username or not password:
        raise ValueError(
            "Thiếu NEO4J_URI, NEO4J_USERNAME hoặc NEO4J_PASSWORD trong file .env"
        )

    driver = GraphDatabase.driver(
        uri,
        auth=(username, password)
    )

    return driver


def test_connection():
    """Kiểm tra kết nối và thực hiện truy vấn Cypher đơn giản."""

    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = get_neo4j_driver()

    try:
        # Kiểm tra khả năng kết nối đến Neo4j
        driver.verify_connectivity()

        # Thực hiện truy vấn thử nghiệm
        with driver.session(database=database) as session:
            result = session.run(
                "RETURN 'DiaCare connected to Neo4j' AS message"
            )

            record = result.single()

            print("Ket noi Neo4j AuraDB thanh cong!")
            print("Ket qua:", record["message"])

    finally:
        # Đóng kết nối sau khi kiểm tra
        driver.close()


if __name__ == "__main__":
    test_connection()
