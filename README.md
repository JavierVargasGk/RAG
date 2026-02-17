RAG Learning Lab 
This project is a hands-on exploration of Retrieval-Augmented Generation (RAG). I built it to understand how to bridge the gap between static documents and Large Language Models (LLMs) using local hardware acceleration.

How it Works
The system follows a standard RAG pipeline but with a custom re-ranking step to improve accuracy:
Ingest: Documents are turned into "embeddings" (vectors) using Voyage AI and stored in a ParadeDB (PostgreSQL) database.
Retrieve: When you ask a question, the system finds the 10 most similar text chunks from the database.
Rerank: A Cross-Encoder model running on my NVIDIA RTX 4060 re-evaluates those 10 chunks to pick the top 3 most relevant ones.
Generate: The best context is sent to Ollama (Llama 3.1) to generate a final, grounded answer.

The Setup (Hybrid Architecture)
To learn how to manage distributed services, I split the project across three environments:
WSL2 (Linux): Runs the Python logic, the Chainlit UI, and the database.
Windows Host: Runs Ollama and the GPU drivers to give the models direct access to the RTX 4060 VRAM.
MacBook: My primary development interface, connected to the PC via VS Code Remote-SSH.


Tech Stack
Models: Llama 3.1, Cross-Encoders, Voyage.
Storage: ParadeDB / PostgreSQL.
UI: Chainlit.

Tools: Python, WSL2, Docker, and Git.
