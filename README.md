# 💄 Beauty Lakehouse Analytics Platform — Developer Guide

A complete **multi-model data engineering platform** for a beauty e-commerce dataset.

This project ingests, cleans, transforms, enriches, and distributes data across **Delta Lake, MongoDB Atlas, a Parquet Warehouse, and Neo4j AuraDB**, using a **hybrid Databricks + local execution workflow**.

This guide is written as a **developer-focused tutorial** so any team member can understand, run, and extend the platform.

---

# 📁 Project Structure

```
BEAUTY_LAKEHOUSE/
│
├── .github/
│
├── data/
│   └── raw/
│       ├── customers.csv
│       ├── metadata.json
│       ├── order_items.csv
│       ├── orders.csv
│       └── products.csv
│
├── notebooks/
│   ├── config/
│   ├── utils/
│   ├── 00_environment_setup.ipynb
│   ├── 01_dataLake_ingestion.ipynb
│   ├── 02_document_db_mongodb.ipynb
│   ├── 03_data_warehouses_analytics.ipynb
│   └── 04_graph_db.ipynb
│
├── scripts/
│   ├── neo4j_local.py
│   └── validate_dataset.py
│
├── src/
│   └── generate_data.py
│
├── .gitignore
└── README.md
```

---

# 🏗️ Architecture Overview

This project follows a **multi-model Lakehouse architecture**, where each database serves a specific purpose.

```
GitHub (raw CSV)
        ↓
Notebook 00 — Environment Setup
        ↓
Notebook 01 — Data Lake (Delta Lake)
        ↓
 ┌────────────────┬────────────────┬──────────────────────────────┐
 │                │                │                              │
Notebook 02    Notebook 03     Notebook 04 (Cypher Builder)     Local Script
MongoDB         Warehouse        MongoDB Bridge                  neo4j_local.py
```

### 🔹 Delta Lake (Databricks)

Stores **raw and curated Delta tables**.

### 🔹 MongoDB Atlas

Stores **rich product documents** and acts as a **bridge for Neo4j queries**.

### 🔹 Parquet Warehouse

Stores **analytics-ready fact tables and KPIs**.

### 🔹 Neo4j AuraDB

Stores **graph relationships** between:

- Customers  
- Orders  
- Products  

---

# 🧰 Notebook 00 — Environment Initialization

This notebook prepares the **shared Unity Catalog environment** used by all team members.

It must be executed **once by the project owner or workspace admin**.

## Objects Created

### Catalog

```
beauty_catalog
```

Top-level namespace for all project data.

### Schemas

```
curated      → cleaned Delta tables  
warehouse    → fact & dimension tables  
analytics    → KPI and reporting outputs
```

### Volume

```
/Volumes/workspace/beauty/data
```

Shared storage for:

- Raw files  
- Curated Delta outputs  
- Warehouse exports  

---

# 🚀 Running the Full Pipeline

## Step 0 — Notebook 00: Environment Setup

Initializes:

- Catalog  
- Schemas  
- Volume  
- Shared paths  

---

## Step 1 — Notebook 01: Data Lake Ingestion

- Loads raw CSV files  
- Cleans and standardizes schemas  
- Writes curated Delta tables  

---

## Step 2 — Notebook 02: MongoDB Layer

Connects to **MongoDB Atlas**.

Actions:

- Creates product documents  
- Inserts curated product data  

---

## Step 3 — Notebook 03: Warehouse Layer

Creates analytics tables including:

- `fact_sales`
- Revenue KPIs
- Customer metrics
- Product performance metrics

---

## Step 4 — Notebook 04: Graph Layer (Cypher Builder)

Loads curated Delta tables.

Samples:

- customers  
- orders  
- products  

Builds Cypher queries for:

- Customer nodes  
- Product nodes  
- Order nodes  
- Relationships  

```
PLACED
CONTAINS
PURCHASED
```

Outputs:

- Saves queries + row data to MongoDB  
- Saves analysis queries  

---

## Step 5 — Local Script Execution

Run:

```bash
python scripts/neo4j_local.py
```

This script:

1. Reads Cypher queries from MongoDB  
2. Connects to Neo4j AuraDB  
3. Clears the graph  
4. Creates nodes and relationships  
5. Runs analysis queries  
6. Saves results back to MongoDB  

---

# 🧩 Utility Module — `load_curated_tables.py`

This module provides helper functions for loading **curated Delta tables** and validating them.

### Key Functions

```python
load_customers(spark)
load_products(spark)
load_orders(spark)
load_order_items(spark)
load_all_curated_tables(spark)
validate_table(df, name)
```

Used primarily in:

- Notebook 03 — Warehouse  
- Notebook 04 — Graph Database  

---

# ⚙️ Configuration Module — `settings.py`

Defines all shared paths and Unity Catalog references.

### Volume Paths

```
/Volumes/workspace/beauty/data/curated/*
/Volumes/workspace/beauty/data/warehouse/*
```

### Unity Catalog Tables

```
beauty_catalog.curated.customers
beauty_catalog.curated.products
beauty_catalog.curated.orders
beauty_catalog.curated.order_items
beauty_catalog.warehouse.fact_sales
```

This ensures **all team members load data from the same governed tables**.

---

# 🧠 Graph Model (Neo4j)

### Nodes

```
(:Customer)
(:Order)
(:Product)
```

### Relationships

```
(:Customer)-[:PLACED]->(:Order)
(:Order)-[:CONTAINS]->(:Product)
(:Customer)-[:PURCHASED]->(:Product)
```

### Why not store full product details in Neo4j?

MongoDB already stores **rich product documents**.

Neo4j only needs:

```
product_id
name
category
```

This avoids duplication and keeps each database focused on its strengths.

---

# 📊 Warehouse KPIs

Generated in **Notebook 03**:

- Total revenue  
- Revenue by category  
- Revenue by month  
- Orders per customer  
- Top-selling products  
- Customer activity metrics  

Stored as **Parquet tables** for BI tools.

---

# 🛠️ Tech Stack

```
Databricks
Apache Spark
Delta Lake
MongoDB Atlas
Neo4j AuraDB
Python
Pandas
PyMongo
Neo4j Python Driver
Parquet
```

---

# 🎯 Project Goals

```
✔ Build a multi-model data architecture
✔ Use Delta Lake for ingestion + curation
✔ Use MongoDB for document storage
✔ Use Parquet for analytics
✔ Use Neo4j for relationship modeling
✔ Build Cypher queries programmatically
✔ Execute graph creation locally
✔ Produce business-ready KPIs
```

