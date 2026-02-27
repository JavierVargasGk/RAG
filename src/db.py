import os
import voyageai
import psycopg
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

EMBEDDING_MODEL = "voyage-code-3"

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API"), timeout=120)


def get_connection_string():
    full_url = os.getenv("DATABASE_URL")
    if full_url:
        return full_url
    creds = {
        "user": os.getenv("DB_USER"),
        "pass": os.getenv("DB_PASS"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "name": os.getenv("DB_NAME")
    }
    if not all(creds.values()):
        raise ValueError("Missing database environment variables!")
    
    return f"postgresql://{creds['user']}:{creds['pass']}@{creds['host']}:{creds['port']}/{creds['name']}"

def file_exists(filename: str) -> bool:
    try:
        with psycopg.connect(get_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS(SELECT 1 FROM doc_chunks WHERE filename = %s)", (filename,))
                return cur.fetchone()[0] 
    except Exception as e:
        logger.error(f"Database error while checking file: {e}")
        return False

def embed_text(texts: list, is_query: bool = False):
    """
    Handles all embeddings for the app.
    is_query=True: Optimized for search questions.
    is_query=False: Optimized for document storage.
    """
    input_type = "query" if is_query else "document"
    try:
        res = vo.embed(texts, model=EMBEDDING_MODEL, input_type=input_type,output_dimension=1024)
        return res.embeddings
    except Exception as e:
        logger.error(f"Embedding failure ({EMBEDDING_MODEL}): {e}")
        raise e

def delete_file_from_db(filename: str):
    try:
        with psycopg.connect(get_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM doc_chunks WHERE filename = %s", (filename,))
                conn.commit()
                logger.info(f"Deleted chunks for: {filename}")
    except Exception as e:
        logger.error(f"Error deleting {filename}: {e}")
        