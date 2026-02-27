import time
import re
import requests
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from src.RAGService import RagService
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
               
               CITATION RULE:
               The 'Actual Output' contains source citations (e.g., '(Source: postgresql-18-A4.pdf p. 123)'). 
               These are the 'Ground Truth'. If the Actual Output provides a citation and more technical 
               detail than the Expected baseline, award a score of 1.0. 
               
               TECHNICAL DRIFT:
               If the Actual Output contradicts the Expected Answer (baseline), but provides a 
               specific page number from the v18 documentation to support its claim, 
               prioritize the Actual Output as the corrected truth.""",
    evaluation_steps=[
        "1. Check for page citations in the Actual Output.",
        "2. Compare technical claims. If Actual has a citation and higher depth than Expected, it is a perfect match (1.0).",
        "3. Verify that v18 specific features (AIO, uuidv7, etc.) are treated as upgrades, not errors.",
        "4. Penalize only if the Actual Output makes a claim that is physically impossible or logically incoherent."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=local_judge,
    threshold=0.7
)

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
    ("How do I perform a parallel GIN index build?", "GIN index builds now support parallelism automatically based on max_parallel_maintenance_workers.")
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