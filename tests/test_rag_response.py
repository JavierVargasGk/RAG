import time
import re
import requests
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from src.RAGService import RagService
from src.ingest import getChunks, makeBatches
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("rag_test.log",mode='w'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


class MyJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
    def load_model(self): return self.model_name
    def get_model_name(self): return self.model_name
    
    def generate(self, prompt: str) -> str:
        url = "http://host.docker.internal:11434/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload, timeout=30)
            return response.json().get("response", "")
        except Exception as e:
            return f"Judge Connection Error: {str(e)}"
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
rag_service = RagService()
local_judge = MyJudge()  

correctness_metric = GEval(
    name="Postgres Version Delta & Correctness",
    criteria="""You are a senior technical auditor.
    
               TRUTH RANKING:
               1. Highest (1.0): Actual Output provides a specific citation (Source: [file] p. [page]) 
                  and matches or exceeds the technical depth of the Expected Answer.
               2. Mid-High (0.85): Actual Output provides a specific citation and is technically 
                  accurate, even if it is more concise than the Expected Answer.
               3. Low (0.0 - 0.4): Actual Output makes claims without citations or contradicts 
                  the provided retrieval context.
               
               CITATION RULE:
               The 'Actual Output' contains source citations (e.g., '(Source: postgresql-18-A4.pdf p. 123)'). 
               These are the 'Ground Truth'. If the Actual Output provides a citation and more technical 
               detail than the Expected baseline, award a score of 1.0. 
               
               TECHNICAL DRIFT:
               If the Actual Output contradicts the Expected Answer (baseline), but provides a 
               specific page number from the v18 documentation to support its claim, 
               prioritize the Actual Output as the corrected truth.""",
    evaluation_steps=[
        "1. Verify page citations (Source: filename.pdf p. ###) in the Actual Output.",
        "2. Perfect Match (1.0): Actual is cited and matches or exceeds Expected detail.",
        "3. Concise Accuracy (0.85): Actual aligns with Expected and has a citation, even if less detailed.",
        "4. Version Check: Ensure v18 features (AIO, io_uring, uuidv7) are recognized as valid upgrades.",
        "5. Hallucination Check: Score 0.0 if claims are physically impossible or not in retrieval context."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=local_judge,
    threshold=0.7
)
##Funtion testing
def test_get_chunks_logic():
    text = "ABCDEFGHIJ" # 10 chars
    chunks = getChunks(text, chunkSize=5, overlap=2)
    
    assert len(chunks) == 3
    assert chunks[0] == "ABCDE"
    assert chunks[1] == "DEFGH"
    assert chunks[2] == "GHIJ"
    
def test_get_chunks_empty_input():
    with pytest.raises(ValueError, match="Data cannot be empty"):
        getChunks("")

def test_make_batches_logic():
    """Verify generator yields correct batch sizes."""
    items = list(range(10))
    batches = list(makeBatches(items, batchSize=3))
    
    assert len(batches) == 4 # [0,1,2], [3,4,5], [6,7,8], [9]
    assert len(batches[0]) == 3
    assert len(batches[-1]) == 1
    
def test_check_null_bytes_in_db():
    """
    Validation: No \x00. 
    """
    conn_str = get_connection_string()
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM doc_chunks WHERE content LIKE '%\x00%';")
            count = cur.fetchone()[0]
            assert count == 0, "Found illegal null bytes in the database content!"
            
# RAG Response Testing
@pytest.mark.parametrize("query, expected_answer", [
    ("What are the three supported values for the new io_method parameter?", "The supported values are 'sync', 'worker', and 'io_uring'."),
    ("How do I enable the io_uring I/O method in v18?", "Set 'io_method = io_uring' in postgresql.conf (requires Linux and liburing support)."),
    ("What does the io_workers parameter control?", "It sets the number of background I/O worker processes when io_method is set to 'worker'."),
    ("Are data checksums enabled by default in PostgreSQL 18?", "Yes, new clusters created with initdb now enable data checksums by default."),
    ("What is the major change to NOT NULL constraints in v18?", "NOT NULL constraints now have entries in pg_constraint and can be added as NOT VALID."),
    ("How do I use the new built-in function for timestamp-ordered UUIDs?", "Use the uuidv7() function."),
    ("What is the 'streaming = parallel' option in v18 logical replication?", "It allows the subscriber to apply large in-progress transactions using multiple parallel workers, controlled by the max_parallel_apply_workers_per_subscription parameter."),
    ("What is the purpose of the idle_replication_slot_timeout parameter?", "It automatically cancels inactive replication slots after a specified timeout period."),
    ("Can I use OAuth 2.0 for authentication in Postgres 18?", "Yes, PostgreSQL 18 introduces native support for OAuth 2.0 authentication."),
    ("What is a 'B-tree Skip Scan' in v18?", "An optimization that allows using an index even when the leading column is not specified in the query."),
    ("How do Virtual Generated Columns behave in v18?", "They compute values on-demand at query time and are the new default for generated columns."),
    ("What new system view provides detailed block-level I/O statistics?", "The pg_stat_io view."),
    ("Can pg_upgrade preserve planner statistics in version 18?", "Yes, pg_upgrade can now carry over planner statistics to the new cluster."),
    ("What does the --swap flag do in pg_upgrade?", "It speeds up upgrades by swapping data directories instead of copying or linking files."),
    ("How do I perform a parallel GIN index build?", "GIN index builds now support parallelism automatically based on max_parallel_maintenance_workers."),
    ("What is the new syntax for generic classes in Python 3.12?", "Python 3.12 introduces a compact syntax for generic classes using square brackets, like 'class Group[T]:'."),
    ("How does the new 'type' statement work in 3.12?", "The 'type' statement creates type aliases that are lazily evaluated, for example: 'type Point = tuple[float, float]'."),
    ("What is the benefit of the new f-string parsing in 3.12?",  "F-strings are no longer restricted by the quotes used; you can now reuse quotes, use multi-line expressions, and include backslashes/comments inside f-string expressions."),
    ("What is the difference between a 'list' and a 'tuple' according to the docs?", "Lists are mutable sequences, typically used to store collections of homogeneous items. Tuples are immutable sequences, often used to store heterogeneous data."),
    ("How do I use 'itertools.islice' to get a slice of an iterator?", "islice(iterable, stop) or islice(iterable, start, stop, step) returns an iterator that yields selected elements from the iterable without loading the whole thing into memory."),
    ("What does 'sys.path' represent in Python?",      "A list of strings that specifies the search path for modules. It is initialized from the PYTHONPATH environment variable and installation-dependent defaults."),
    ("How do I securely join file paths using the standard library?", "Use 'os.path.join()' or the 'pathlib' module (e.g., Path('dir') / 'file') to ensure cross-platform compatibility."),    ("What is the purpose of the 'contextlib.contextmanager' decorator?", "It allows you to define a factory function for a 'with' statement context manager without needing to create a full class with __enter__ and __exit__ methods."),
    ("What is a 'coroutine' in Python's asyncio?", "A coroutine is a specialized generator-based function defined with 'async def' that can be suspended and resumed, used for non-blocking I/O operations."),
    ("How do I run an async function as the main entry point in 3.12?", "Use 'asyncio.run(main())', which handles the event loop lifecycle, including creation and shutdown."),
    ("What is 'ExceptionGroup' introduced in Python 3.11/3.12?", "A way to raise multiple unrelated exceptions simultaneously, typically used with 'except*' to handle concurrent errors."),
    ("How do I use a 'finally' block in a try-except statement?", "The 'finally' block is executed no matter what, whether an exception was raised or not, making it ideal for cleaning up resources like files.")
])
def test_rag_response(query, expected_answer):
    start_time = time.perf_counter()
    
    result = rag_service.get_response_and_context(query)
    if result is None or result[0] is None:
        pytest.fail(f"Retriever failure for: {query}")
        
    generator, retrieval_context = result
    actual_output_raw = "".join([token for token in generator])
    #keep the citations for LLM as Judge debugging 
    clean_output = actual_output_raw.strip()
    
    total_latency = time.perf_counter() - start_time
    logger.info(f"\n[LATENCY] {total_latency:.4f}s")
    logger.info(f"[CLEAN OUTPUT]: {clean_output}")

    test_case = LLMTestCase(
        input=query,
        actual_output=clean_output, 
        expected_output=expected_answer,
        retrieval_context=retrieval_context
    )
    correctness_metric.measure(test_case)
    
    latency = time.perf_counter() - start_time
    logger.info("-" * 40)
    logger.info(f"QUERY: {query}")
    logger.info(f"LATENCY: {latency:.2f}s")
    logger.info(f"SCORE: {correctness_metric.score}")
    logger.info(f"REASON: {correctness_metric.reason}")
    
    if correctness_metric.score < 0.7:
        logger.warning(f"LOW SCORE DETECTED. Actual Output: {clean_output}")
    
    assert_test(test_case, [correctness_metric])
    
    #run command
    #deepeval test run tests/test_rag_response.py -n 1 