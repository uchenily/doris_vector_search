# Doris Vector Search Python SDK

## Introduction

A Python SDK of Doris Vector Search for performing vector search operations on Apache Doris database. It provides an easy-to-use interface for table management, index management and performing vector search.

## Features

- Automatically detect schema when creating tables
- Insert vector data efficiently using Stream Load API (currently, we support CSV and Arrow formats)
- Support for various data formats (pandas DataFrame, PyArrow Table, list of dicts)
- Support data and schema validation

## Installation

```shell
pip install doris-vector-search
```

Install in local development mode:

```shell
git clone https://github.com/uchenily/doris_vector_search && cd doris_vector_search
pip install -e .
```

## Basic Usage

### Creating A Client

```python
from doris_vector_search import DorisVectorClient, AuthOptions

# Usage doris default auth options
client = DorisVectorClient("test_database")


# With custom auth options
auth = AuthOptions(
    host="localhost",
    query_port=9030,
    http_port=8030,
    user="root",
    password=""
)
client = DorisVectorClient("test_database", auth_options=auth)
```

### Creating A Table With Data

```python
import pandas as pd

# Test data (pd.DataFrame)
data = pd.DataFrame([
    {"id": 1, "vector": [0.9, 0.4, 0.8], "text": "knight"},
    {"id": 2, "vector": [0.8, 0.5, 0.3], "text": "ranger"},
    {"id": 3, "vector": [0.5, 0.9, 0.6], "text": "cleric"},
])

# Test data (List of dicts)
test_data2 = [
    {'id': 1, 'name': 'Alice', 'embedding': [1.1, 2.2, 3.3]},
    {'id': 2, 'name': 'Bob', 'embedding': [4.4, 5.5, 6.6]},
    {'id': 3, 'name': 'Charlie', 'embedding': [8.8, 9.9, 10.0]},
    {'id': 4, 'name': 'David', 'embedding': [10.1, 11.2, 12.3]},
    {'id': 5, 'name': 'Eve', 'embedding': [15.6, 16.7, 17.8]},
]

# Create table with vector index
table = client.create_table("test_vector_table", data, create_index=True)

# Create table with specific index options
index_options = IndexOptions(index_type="hnsw", metric_type="l2_distance")
table = client.create_table("test_vector_table", data, index_options=index_options)
```

### Adding Data To Existed Table

```python
# Open a existed table
table = client.open_table("test_vector_table")

# Add more data
new_data = pd.DataFrame([
    {"id": 4, "vector": [0.3, 0.8, 0.7], "text": "rogue"},
    {"id": 5, "vector": [0.2, 1.0, 0.5], "text": "thief"},
])
table.add(new_data)

# Add data with specific load options
load_options = LoadOptions(format="csv", batch_size=10000)
table.add(new_data, load_options=load_options)
```


### Vector Search

```python
query_vector = [0.8, 0.3, 0.8]
results = table.search(query_vector).limit(10).to_pandas()
print(results)
```


### Advanced Search Options

```python
results = table.search(query_vector)\
    .limit(5)\
    .distance_range(upper_bound=1.0)\
    .where("text = 'knight'")\
    .select(["id", "text"])\
    .to_pandas()

print(results)
```

### Index Management

```python
from doris_vector_search import IndexOptions

# Create custom index options
index_options = IndexOptions(
    index_type="hnsw",
    metric_type="l2_distance",
    dim=64
)

# Add index
table.add_index(index_options)

# Drop index
table.drop_index()
```

### Setting Session Variables

```python
from doris_vector_search import DorisVectorClient

db = DorisVectorClient(database="test")

# Set session variables
db.with_session("parallel_pipeline_task_num", 1)\
  .with_session("num_scanner_threads", 1)\
  .with_session("enable_profile", False)

# or
db.with_sessions(
    {"parallel_pipeline_task_num": 1, "num_scanner_threads": 1, "enable_profile": False})
```
