# ALBERT: RAG Learning Lab & Investigation Assistant

This project is a hands-on exploration of **Retrieval-Augmented Generation (RAG)**

## Technical Architecture

The system implements a RAG pipeline optimized for accuracy and contextual relevance:

1.  **Ingestion & Embedding:** Documents are vectorized using **Voyage AI** and indexed in a **ParadeDB (PostgreSQL)** database.
2.  **Hybrid Retrieval:** The system retrieves the top 10 most similar chunks based on semantic similarity.
3.  **Local GPU Reranking:** To eliminate noise, a **Cross-Encoder** model runs locally. It re-evaluates the initial 10 chunks to isolate the top 5 most relevant segments.
4.  **Inference:** The filtered context is fed to **Ollama (Llama 3.1)** to generate a grounded, hallucination-free response.



## Distributed Setup (Hybrid Infrastructure)

To simulate production-grade resource management, the project is distributed across a three-tier environment:

* **WSL2 (Linux Subsystem):** Orchestrates the Python logic, the **Chainlit UI**, and the **ParadeDB** instance.
* **Windows Host:** Manages **Ollama** and GPU drivers, providing the models with direct access to **RTX 4060 VRAM** for low-latency inference.
* **Remote Development:** Managed via **VS Code Remote-SSH** from a MacBook, simulating a remote server management workflow.


## Tech Stack

* **Models:** Llama 3.1 (Inference), Cross-Encoders (Reranking), Voyage AI (Embeddings).
* **Data Layer:** ParadeDB / PostgreSQL.
* **Interface:** Chainlit (Customized with GitHub-Dark CSS & LaTeX support).
* **Tools:** Python 3.10, WSL2, Docker, Git.


## Key Engineering Takeaways
* **Hardware Optimization:** Successfully offloaded compute-heavy tasks (Reranking/Inference) to local GPU hardware.
* **Search Precision:** Improved retrieval accuracy by implementing a "Retrieve & Rerank" strategy rather than relying on raw vector similarity.
* **Environment Management:** Configured cross-platform communication between Linux (WSL2) and Windows for high-performance AI workloads.
