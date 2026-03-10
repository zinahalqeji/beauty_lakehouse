"""
04_neo4j_local.py
=================
Beauty Lakehouse Analytics Platform
-------------------------------------
This script runs locally (VS Code or PyCharm) to:
  1. Connect to MongoDB Atlas and retrieve the Cypher queries built in Notebook 04
  2. Connect to Neo4j Aura
  3. Execute the queries to build the graph (nodes + relationships)
  4. Run analysis queries and print the results
  5. Save analysis results back to MongoDB for use in Databricks

Requirements (install before running):
  pip install neo4j pymongo certifi python-dotenv

Credentials are loaded automatically from the .env file.
Make sure your .env file contains:
  MONGO_USER=...
  MONGO_PASS=...
  MONGO_URI=mongodb+srv://<user>:<pass>@<your-cluster>.mongodb.net/
"""

import ssl
import certifi
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

load_dotenv()

# MongoDB – var och en har sin egen URI i .env
MONGO_URI = os.getenv("MONGO_URI")

# Neo4j – delade uppgifter hårdkodade
NEO4J_URI  = "neo4j+s://26f6a455.databases.neo4j.io"
NEO4J_USER = "26f6a455"
NEO4J_PASS = "Rsc_8H8hIvpPjuMg-UJLcV8-nzTVd6XvF8PEHEyLX5Y"

print("=" * 60)
print("STEP 1 – Connecting to MongoDB Atlas...")
print("=" * 60)

mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = mongo_client["beauty_lakehouse_db"]
mongo_client.admin.command("ping")
print("✓ Connected to MongoDB Atlas!\n")

print("=" * 60)
print("STEP 2 – Connecting to Neo4j Aura...")
print("=" * 60)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
driver.verify_connectivity()
print("✓ Connected to Neo4j Aura!\n")

print("=" * 60)
print("STEP 3 – Clearing existing graph data...")
print("=" * 60)

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
print("✓ Existing graph data cleared.\n")

print("=" * 60)
print("STEP 4 – Loading Cypher queries from MongoDB and executing against Neo4j...")
print("=" * 60)

build_queries = list(db.neo4j_queries.find())

for doc in build_queries:
    query_name = doc["query_name"]
    cypher     = doc["cypher"]
    rows       = doc["rows"]

    print(f"\nRunning: {query_name} ({len(rows)} rows)...")

    with driver.session() as session:
        session.run(cypher, rows=rows)

    print(f"  ✓ {query_name} completed.")

print("\n✓ All graph nodes and relationships created successfully!\n")

print("=" * 60)
print("STEP 5 – Running Graph Analysis Queries...")
print("=" * 60)

analysis_queries = list(db.neo4j_analysis_queries.find())
analysis_results = []

for doc in analysis_queries:
    query_name  = doc["query_name"]
    description = doc["description"]
    cypher      = doc["cypher"]

    print(f"\n--- {description} ---")

    with driver.session() as session:
        result  = session.run(cypher)
        records = [dict(r) for r in result]

    for record in records:
        print("  ", record)

    analysis_results.append({
        "query_name" : query_name,
        "description": description,
        "results"    : records
    })

print("\n" + "=" * 60)
print("STEP 6 – Saving analysis results back to MongoDB...")
print("=" * 60)

db.neo4j_results.drop()
db.neo4j_results.insert_many(analysis_results)

print("✓ Analysis results saved to MongoDB collection: neo4j_results")
print("\nThese results can now be retrieved in Databricks for reporting.\n")

driver.close()
mongo_client.close()

print("=" * 60)
print("All done! Graph built and analysis complete.")
print("=" * 60) 