
import requests
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval, FaithfulnessMetric,ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from src.RAGService import RagService

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
    name="Postgres Technical Correctness",
    criteria="Determine if the response accurately reflects PostgreSQL 16+ syntax and administrative best practices.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    evaluation_steps=[
        "Compare the core technical facts in the actual output against the expected output.",
        "Verify that any SQL commands or configuration names are accurate.",
        "Check for any major technical hallucinations."
    ],
    model=local_judge,
    threshold=0.7
)
@pytest.mark.parametrize("query, expected_answer", [
    (
        "How do I change the port in postgresql.conf?", 
        "Edit the 'port' parameter in postgresql.conf (default 5432) and restart the server."
    ),
    (
        "What is the command to create a physical replication slot?", 
        "SELECT * FROM pg_create_physical_replication_slot('slot_name');"
    )
])
def test_rag_response( expected_answer):
    actual_output = "".join([t for t in rag_service.get_response(query)])
    test_data = LLMTestCase(   
        input=query,
        actual_output=actual_output,
        expected_output=expected_answer
        )
    assert_test(test_data, [correctness_metric])
    
def test_rag_robustness():
    query = "What is the command to create a physical replication slot?"
    ntext(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    # Run both at once
    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevancy = ContextualRelevancyMetric(threshold=0.7)
    
    assert_test(test_case, [faithfulness, relevancy])
    