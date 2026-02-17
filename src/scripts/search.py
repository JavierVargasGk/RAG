from dotenv import load_dotenv
import os
import voyageai
from sentence_transformers import CrossEncoder
import chainlit as cl
import requests
import psycopg

import sys
from pathlib import Path
# 1. Calculamos la ruta raíz (RAG/) de forma dinámica
# Subimos 3 niveles desde src/scripts/search.py para llegar a RAG/
root_path = Path(__file__).resolve().parents[2] 

# 2. Insertamos esa ruta al principio de la lista de búsqueda de Python
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# 3. AHORA sí podemos importar tu archivo db.py
try:
    from db import get_connection_string
    print(f"✅ Conexión con db.py exitosa desde: {root_path}")
except ImportError as e:
    print(f"❌ Error: No se encontró db.py en {root_path}")
    print(f"Path actual de búsqueda: {sys.path[0]}")
    sys.exit(1)
    
    

load_dotenv()
vo = voyageai.Client(os.getenv("VOYAGE_API"))

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
    response = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.1", "prompt": prompt, "stream": False},timeout=30)
    
    
    await cl.Message(content=response.json()['response']).send()
    