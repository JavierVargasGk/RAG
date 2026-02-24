
from dotenv import load_dotenv
import os
import voyageai
from sentence_transformers import CrossEncoder
import chainlit as cl
import requests
import subprocess
import json
import psycopg
import logging
from pathlib import Path
from core.db import get_connection_string, embed_text

logger = logging.getLogger(__name__)
load_dotenv()
vo = voyageai.Client(os.getenv("VOYAGE_API"))

def get_ollama_endpoint():
    try:
        window_ip = subprocess.check_output("ip route | grep default | awk '{print $3}'", shell=True).decode().strip()
        logger.info(f"Detected Windows Host IP: {window_ip}")
        test_url = f"http://{window_ip}:11434/api/tags"
        requests.get(test_url, timeout=1)
        return f"http://{window_ip}:11434"
    except:
        return f"http://localhost:11434"

OLLAMA_BASE = get_ollama_endpoint()


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

@cl.on_message
async def main(message: cl.Message):
    try:
        query_vectors = embed_text([message.content], is_query=True)
        query_vector = query_vectors[0]
    except Exception as e:
        logger.error(f"Search failed at embedding stage: {e}")
        await cl.Message(content="Error: Could not process query. Please try again.").send()
        return
            
    conn_str = get_connection_string()
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content, filename, page_number, 
                    (paradedb.score(id) + (1.0 - (embedding <=> %s::vector))) as combined_score
                FROM doc_chunks 
                WHERE content @@@ %s  
                OR embedding <=> %s::vector < 0.5 
                ORDER BY combined_score DESC
                LIMIT 20
            """, (query_vector, message.content, query_vector))
            candidates = cur.fetchall()
    if not candidates:
        await cl.Message(content="No relevant documents found.").send()
        return
    
    # 2. Reranking 
    pairs = [[message.content, row[0]] for row in candidates]
    scores = reranker.predict(pairs)
    ranked_results = [row for _, row in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
    
    context_parts = []
    source_elements = []
    
    # Context
    for i, (content, filename, page, score) in enumerate(ranked_results[:5]):

        label = f"Ref_{i+1}_{filename}_p{page}"
        context_parts.append(f"SOURCE: {label}\nCONTENT: {content}")
        source_elements.append(cl.Text(content=content, name=label, display="side"))
    context_string = "\n\n---\n\n".join(context_parts)
    prompt = fr"""
You are a precise technical research assistant and documentation referencer.

Your goal is to answer strictly and exclusively using the provided Context.

==============================
STRICT RULES
==============================

1. INTERNAL KNOWLEDGE LIMIT:
Use ONLY information explicitly stated in the Context.

2. NO SYNTHESIS:
Do NOT generalize or add commentary.

3. MISSING INFORMATION:
If the answer is not in the Context, respond exactly with:
"Information not found in provided documents."

4. CITATION PER CLAIM:
Include citations in parentheses, e.g., (Source: filename, p. 89).

5. MATHEMATICAL FORMATTING:
Only use LaTeX for actual mathematical symbols, variables, and formulas. 
Do NOT wrap regular text or explanations in LaTeX \text{{}} blocks.

6. INLINE MATH:
Wrap inline math using single dollar signs with no spaces. Example: $x^2$.

7. BLOCK MATH:
Use $$ only for standalone formulas on their own lines.
Correct example:

$$
\\sum_{{n=0}}^{{\\infty}} a_n x^n
$$

8. DO NOT SIMPLIFY:
Do not rewrite results unless the Context explicitly shows that step.

9. NO DOUBLE FORMATTING:
Output math ONLY in LaTeX format. Do not provide a "plain text" or "simplified" version of the equation immediately after the LaTeX block.

10. NO REPETITION: 
Do not provide a text-based or "simplified" version of a mathematical formula after providing the LaTeX version. Provide the LaTeX version ONLY.

==============================
Context:
{context_string}

==============================
Question:
{message.content}

==============================
Answer (with citations):
"""
    
    
    # 4. Response to UI
    msg = cl.Message(content="", elements=source_elements)
    await msg.send()
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {"model": "llama3.1", "prompt": prompt, "stream": True, "options": {"num_ctx": 8192}}

    with requests.post(url, json=payload, stream=True, timeout=60) as response:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if not chunk.get("done"):
                    token = chunk.get('response', '')
                    await msg.stream_token(token)
    
    await msg.update()
