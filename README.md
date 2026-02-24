# ALBERT 
**RAG with Hybrid Search & Local GPU Reranking**

## Quick Start
1. **Clone the Repo:** `git clone https://github.com/JavierVargasGk/RAG`
2. **Infrastructure:** Start the docker container that holds the database (ParadeDB): `docker-compose up -d`
3. **Engine:** Install and run **Ollama** on your host machine:
   ```powershell
   # Windows (PowerShell)
   irm [https://ollama.com/install.ps1](https://ollama.com/install.ps1) | iex
4. create and open the venv using `python -m venv venv` then activate with `.\venv\Scripts\activate`(Windows) or `source /venv/bin/activate`(Mac/Linux)
5. Install dependencies with `pip install -r requirements.txt`
6. Make a .env file with your Voyage API key, and the docker information, then save it.
7. Now you can add whatever files you want the RAG to work with into your `data/` folder.
8. Finally you just run the main script `app.py` and once its done ingesting the data you uploded (can take a while if your data is too big), it should auto open your very own RAG.
   
## Future TODO
* **Prettier front-end**: GUI is a work in progress.
* **Web based ingestion**: I want the RAG to be able to ingest files inside of the browser GUI.
* **Better ingesting for library/framework documentations**: Few are the libraries that have their entire framework/library docs on a PDF, so adding this is a priority if this is meant to be used as a coding helper.

## Project Architecture
The system implements a production-grade RAG pipeline focused on high-precision retrieval:
1. **Ingestion & Embedding:** Automated ETL pipeline using **Voyage** (specialized for technical data) with vector indexing in **ParadeDB**.
2. **Hybrid Retrieval:** Executes a fused search query (BM25 + Vector Similarity) to capture both keyword exact-matches and semantic context.
3. **Local GPU Reranking:** Implements a **Cross-Encoder (ms-marco-MiniLM)** pass on the top 10 candidates to mitigate retrieval "noise" and ensure only the top 5 highly-relevant chunks reach the LLM.
4. **Grounded Inference:** Context-window grounding via **Llama 3.1**, enforced with strict system prompts to prevent hallucinations and ensure source-backed responses.

### Hybrid Infrastructure & Networking
To maximize local hardware while maintaining a Linux-native environment:
* **Linux (WSL2):** Hosts the application logic, **Dockerized ParadeDB**, and the persistence layer.
* **Windows Host:** Serves as the high-performance compute node, hosting **Ollama** and bridging GPU access for the **RTX 4060**.
* **Remote Management:** Developed using a **Headless Server workflow** via **SSH** and **Tailscale**, allowing for full development and monitoring from a remote client.

### Tech Stack
* **Models:** Llama 3.1 (Inference), Cross-Encoders (Reranking), Voyage AI (Embeddings).
* **Data Layer:** ParadeDB / PostgreSQL.
* **Interface:** Chainlit (LaTeX formula support for maths).
* **Tools:** Python, WSL2, Docker, Git.

### Key Engineering Takeaways
* **Hardware Optimization:** Successfully offloaded compute-heavy tasks (Reranking/Inference) to local GPU hardware.
* **Search Precision:** Improved retrieval accuracy by implementing a "Retrieve & Rerank" strategy rather than relying on raw vector similarity.
* **Environment Management:** Configured cross-platform communication between Linux (WSL2) and Windows for high-performance AI workloads.









