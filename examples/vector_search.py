from doris_vector_search import DorisVectorClient

# Create client
db = DorisVectorClient(database="test_database")

table = db.open_table("test_table")
result = table.search([0.5, 0.9, 0.6]).select(["text"]).limit(3).to_arrow()
print(result)
