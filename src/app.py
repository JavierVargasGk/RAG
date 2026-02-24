
import os
from chainlit.cli import run_chainlit
from src.core.ingest import run
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
if __name__ == "__main__":
    run()
    target_path = os.path.join("src", "core", "search.py")
    run_chainlit(target_path)

