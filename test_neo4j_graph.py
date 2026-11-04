import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE")

driver = GraphDatabase.driver(
    uri,
    auth=(username, password)
)

try:
    with driver.session(database=database) as session:

        # Đếm số Entity trong Knowledge Graph
        entity_result = session.run("""
            MATCH (n:Entity)
            RETURN count(n) AS total
        """)
        total_entities = entity_result.single()["total"]

        # Đếm tổng số Relationship
        relationship_result = session.run("""
            MATCH ()-[r]->()
            RETURN count(r) AS total
        """)
        total_relationships = relationship_result.single()["total"]

        print("=== DIACARE KNOWLEDGE GRAPH ===")
        print("Database:", database)
        print("Total entities:", total_entities)
        print("Total relationships:", total_relationships)

finally:
    driver.close()