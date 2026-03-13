import pandas as pd
from doris_vector_search import DorisVectorClient
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Test data
data = pd.DataFrame([
    {"id1": 1, "vector1": [0.9, 0.4, 0.8], "text": "knight"},
    {"id1": 2, "vector1": [0.8, 0.5, 0.3], "text": "ranger"},
    {"id1": 3, "vector1": [0.5, 0.9, 0.6], "text": "cleric"},
    {"id1": 4, "vector1": [0.3, 0.8, 0.7], "text": "rogue"},
    {"id1": 5, "vector1": [0.2, 1.0, 0.5], "text": "thief"},
])

# Create client
db = DorisVectorClient(database="test_database")

table = db.create_table("test_table", data, create_index=False, overwrite=True)
print(f"Table name: {table.table_name}")
print(f"Table schema: {table.schema()}")
