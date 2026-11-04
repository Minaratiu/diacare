import os
from getpass import getpass

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
database = os.getenv("NEO4J_DATABASE")

print("URI:", uri)
print("Username:", username)
print("Database:", database)

password = getpass("Nhap mat khau AuraDB: ")

driver = GraphDatabase.driver(
    uri,
    auth=(username, password)
)

try:
    driver.verify_connectivity()

    # Sử dụng tên database đọc từ file .env
    with driver.session(database=database) as session:
        result = session.run(
            "RETURN 'DiaCare connected to Neo4j' AS message"
        )

        print("KET NOI THANH CONG!")
        print("Ket qua:", result.single()["message"])

except Exception as error:
    print("KET NOI THAT BAI!")
    print("Loai loi:", type(error).__name__)
    print("Chi tiet:", error)

finally:
    driver.close()