import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

from core.ingest import run_ingest
from chainlit.cli import run_chainlit
import os


if __name__ == "__main__":
    logger.info("Starting the RAG Ingestion process...")
    run_ingest()
    target_path = os.path.join("src", "core", "search.py") 
    run_chainlit(target_path)
