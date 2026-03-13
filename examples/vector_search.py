from doris_vector_search import DorisVectorClient
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create client
db = DorisVectorClient(database="test_database")

table = db.open_table("test_table")
result = table.search([0.5, 0.9, 0.6]).select(["text"]).limit(3).to_arrow()
print(result)
result = table.search([0.5, 0.9, 0.6], include_distance=True).select(["text"]).limit(3).to_arrow()
print(result)
