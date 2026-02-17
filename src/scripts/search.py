from dotenv import load_dotenv
import os
import voyageai
from sentence_transformers import CrossEncoder
import chainlit as cl
import requests
import psycopg
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2] 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
try:
    from db import get_connection_string
except ImportError as e:
    sys.exit(1)
    
load_dotenv()
vo = voyageai.Client(os.getenv("VOYAGE_API"))

def get_ollama_endpoint():
    # 1. Check if we are in WSL
    try:
        # This command gets the Gateway IP (the Windows Host)
        window_ip = subprocess.check_output("ip route | grep default | awk '{print $3}'", shell=True).decode().strip()
        # Test if Ollama is reachable on that IP
        test_url = f"http://{window_ip}:11434/api/tags"
        requests.get(test_url, timeout=1)
        return f"http://{window_ip}:11434"
    except:
        # 2. Fallback to localhost (for production or native Linux)
        return "http://localhost:11434"

OLLAMA_BASE = get_ollama_endpoint()


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",device="cuda")


@cl.on_message
async def main(message: cl.Message):
    #Vector and Query
    res = vo.embed([message.content], model="voyage-finance-2")
    query_vector = res.embeddings[0]
    connect_string = get_connection_string()
    with psycopg.connect(connect_string) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM doc_chunks ORDER BY embedding <=> %s::vector LIMIT 10", (query_vector,))
            candidates = [row[0] for row in cur.fetchall()]
            
    if not candidates:
        await cl.Message(content="No relevant documents found.").send()
        return
    
    #Reranking
    pairs = [[message.content, candidate] for candidate in candidates]
    scores = reranker.predict(pairs)
    ranked_candidates = [candidate for _, candidate in sorted(zip(scores, candidates), reverse=True)]
    
    #Ollama
    context = "\n---\n".join(ranked_candidates)
    prompt = f"Context: {context}\n\nQuestion: {message.content}\nAnswer:"
    response = requests.post("http://{OLLAMA_BASE}:11434/api/generate", json={"model": "llama3.1", "prompt": prompt, "stream": False},timeout=30)
    
    
    
    await cl.Message(content=response.json()['response']).send()
    