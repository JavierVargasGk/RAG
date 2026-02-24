import logging
from core.ingest import run_ingest
from chainlit.cli import run_chainlit
import time
import os

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


if __name__ == "__main__":
    logger.info("Starting the RAG Ingestion process...")
    while True:
        try:
            run_ingest()
            logger.info("Ingestion completed successfully!")
            break
        except Exception as e:
            logger.error(f"Ingestion crashed with error: {e}. Retrying in 10s...")
            import time
            time.sleep(10)
            
    target_path = os.path.join("src", "core", "search.py") 
    run_chainlit(target_path)