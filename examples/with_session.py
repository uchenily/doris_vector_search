from doris_vector_search import DorisVectorClient

# Create client
db = DorisVectorClient(database="test")

db.with_session("parallel_pipeline_task_num", 1)\
  .with_session("num_scanner_threads", 1)\
  .with_session("enable_profile", False)

db.with_sessions(
    {
        "parallel_pipeline_task_num": 1,
        "num_scanner_threads": 1,
        "enable_profile": False,
    }
)
