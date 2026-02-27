from src.db import get_connection_string, embed_text
import requests
import json
import subprocess
import psycopg
import logging
from sentence_transformers import CrossEncoder
logger = logging.getLogger(__name__)


class RagService:
    def __init__(self):
        self.OLLAMA_BASE = self.get_ollama_endpoint()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        
    @staticmethod
    def get_ollama_endpoint():
        try:
            window_ip = subprocess.check_output("ip route | grep default | awk '{print $3}'", shell=True).decode().strip()
            return f"http://{window_ip}:11434"
        except:
            return "http://localhost:11434"
    
    #Database search with combined ParadeDB and vector similarity scoring
    def search_database(self, query_vector, query):
        conn_str = get_connection_string()
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                        SELECT content, filename, page_number,
                        ( paradedb.score(id) + (1.0 - (embedding <=> %s::vector))) as combined_score
                        FROM doc_chunks 
                        WHERE 
                            id @@@ paradedb.match('content', %s)
                            OR 
                            embedding <=> %s::vector < 0.5 
                        ORDER BY combined_score DESC
                        LIMIT 20
                        """, 
                        (query_vector, query, query_vector))
                return cur.fetchall()
    
    def rerank_results(self, query, candidates):
        if not candidates: return []
        pairs = [[query,c[0]] for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked_results = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
        return ranked_results[:10]

    
    # RAG response generation with strict adherence to provided context and source citation
    def generate_response(self, query, context):
        prompt = f"""
            You are a Technical Support Engineer. 
            Your goal is to provide high-precision answers based ONLY on the provided context.
            <instructions>
            STRICT RULES:
            1. Use ONLY information from the Context. If missing, say: "Information not found in provided documents."
            2. Cite sources in parentheses: (Source: filename, p. XX). 
            3. DE-DUPLICATION: If multiple sources provide the same fact, combine them into one sentence and list all sources at the end, e.g., (Source: file1, p. 10; file2, p. 55).
            4. Formatting: Use `code blocks` for Code/SQL/parameters. Do NOT use LaTeX for version numbers or simple integers.
            </instructions>
            <context>
            {context}
            </context>

            <question>
            {query}
            </question>"""
            
        url = f"{self.OLLAMA_BASE}/api/generate"
        payload = {"model": "llama3.1", "prompt": prompt, "stream": True, "options": {"num_ctx": 8192}}
        # Stream response from Ollama and yield tokens as they arrive
        full_response = []
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    token = chunk.get("response", "")
                    
                    full_response.append(token)
                    yield token

    # Main method to get RAG response for a query
    def get_response(self, query):
        query_vector = embed_text([query], is_query=True)[0]
        candidates = self.search_database(query_vector, query)
        
        if not candidates:
            yield "No relevant documents found."
            return

        ranked_results = self.rerank_results(query, candidates)
        context = "\n".join([f"Source: {r[1]} p.{r[2]}\nContent: {r[0]}" for r in ranked_results])
        
        yield from self.generate_response(query, context)

  