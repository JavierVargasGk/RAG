import fitz
from core.db import get_connection_string, file_exists, embed_text
import time
import logging
import voyageai
import psycopg
from dotenv import load_dotenv
import pickle
import os


logger = logging.getLogger(__name__)

def getTextFromPDF(filePath: str):
    doc = fitz.open(filePath)
    pages_data = []
    logger.info(f"Parsing {os.path.basename(filePath)} with Markdown-aware logic.")
    for i, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        md_text = ""
        for b in blocks:
            if "lines" not in b: continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text: continue
                    size = span["size"]
                    if size > 14:
                        md_text += f"\n# {text}\n"
                    elif 12 < size <= 14:
                        md_text += f"\n## {text}\n"
                    elif span["flags"] & 2**4:
                        md_text += f" **{text}** "
                    else:
                        md_text += f" {text} "
        clean_text = md_text.replace('\x00', '').replace('\ufb01', 'fi').replace('\ufb02', 'fl')
        pages_data.append({"text": clean_text, "page": i + 1})
        
    doc.close()
    return pages_data

#Split data into chunks of 1000 characters with an overlap of 200 characters (or changed if needed)
def getChunks(data: str,chunkSize: int = 1000, overlap: int=200):
    if(data == ""):
        logger.warning("Empty data received for chunking.")
        raise ValueError("Data cannot be empty.")
    if(chunkSize <= 0):
        logger.error(f"Invalid chunk size: {chunkSize}. Must be greater than 0.")
        raise ValueError("Chunk size must be greater than 0.")
    if(overlap < 0):
        logger.error(f"Invalid overlap: {overlap}. Must be non-negative.")
        raise ValueError("Overlap must be non-negative.")
    if(overlap >= chunkSize):
        logger.error(f"Invalid overlap: {overlap}. Must be less than chunk size: {chunkSize}.")
        raise ValueError("Overlap must be less than chunk size.")
    chunks = []
    for i in range(0, len(data), chunkSize - overlap):
        chunk = data[i:i + chunkSize]
        chunks.append(chunk)
        if(i + chunkSize >= len(data)):
            break
    return chunks
#Turn the chunks into batches of 128 (or changed if needed)  
def makeBatches(chunks, batchSize: int = 128):
    if(batchSize <= 0):
        raise ValueError("Batch size must be greater than 0.")
    for i in range(0, len(chunks), batchSize):
        yield chunks[i:i + batchSize]
        
        
load_dotenv()
#Using voyage cuz free for a while
vo = voyageai.Client(os.getenv("VOYAGE_API"))


def ingestPdf(filePath: str):
    filename = os.path.basename(filePath)
    checkpoint_path = f"data/checkpoints/checkpoint_{filename}.pkl" # 
    if (file_exists(filename)):
        logger.info(f"Skipping {filename}: Already exists.")
        return
    # i hate everything, 90k tokens gone
    #This opens a pkl file, loading all the data if it exists so we dont have to embed it again from 0 but instead keep it there
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
            all_chunks = data["chunks"]
            all_vectors = data["vectors"]
            all_metadata = data["metadata"]     
            start_idx = len(all_vectors)
    else:
    #if pickle file doesnt exist we start from 0 and do the whole process, this is for the first time ingesting a file or if something went wrong and we need to start over
        pages_data = getTextFromPDF(filePath)
        all_chunks = []
        all_metadata = [] 
        start_idx = 0
        all_vectors = []
        for page in pages_data:
            page_chunks = getChunks(page["text"])
            for chunk in page_chunks:
                all_chunks.append(chunk)
                all_metadata.append({"file": filename, "page": page["page"]})
    #Embeds regardless of state, if there is none to embed it wont even get to this part, so its fine.
    SMALL_BATCH_SIZE = 10 
    REQUEST_INTERVAL = 20.5
    logger.info(f"Embedding {len(all_chunks)} chunks...")
    for i in range(start_idx, len(all_chunks), SMALL_BATCH_SIZE):
        start_time = time.time()
        batch = all_chunks[i : i + SMALL_BATCH_SIZE]
        retry_delay = 5
        while True: 
            try:
                embeddings = embed_text(batch, is_query=False)
                all_vectors.extend(embeddings)
                logger.info(f"Progress: {i + len(batch)}/{len(all_chunks)}")
                elapsed = time.time() - start_time
                sleep_needed = max(0, REQUEST_INTERVAL - elapsed)
                logger.info(f"Batch took {elapsed:.2f}s. Sleeping for {sleep_needed:.2f}s")
                time.sleep(sleep_needed)
                if (i + len(batch)) % 100 == 0:
                    with open(checkpoint_path, "wb") as f:
                        checkpoint_data = {
                        "chunks": all_chunks, 
                        "vectors": all_vectors, 
                        "metadata": all_metadata,                                   
                        "last_index": i + len(batch)
                        }
                        pickle.dump(checkpoint_data, f)
                        logger.info(f"Checkpoint saved at {i + len(batch)}")
                break
            except Exception as e:
                if "429" in str(e) or "limit" in str(e).lower():
                    logger.warning(f"Rate limit reached. Retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                    if retry_delay > 120: raise e
                else:
                    logger.error(f"Embedding error: {e}")
                    raise e
    conn_str = get_connection_string()
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                sql = "COPY doc_chunks (content, embedding, filename, page_number) FROM STDIN"
                with cur.copy(sql) as copy:
                    for chunk, vector, meta in zip(all_chunks, all_vectors, all_metadata):
                        vector_str = "[" + ",".join(map(str, vector)) + "]"
                        copy.write_row((chunk, vector_str, meta["file"], meta["page"]))
                        
        logger.info(f"Successfully ingested {len(all_chunks)} chunks.")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            
    except Exception as e:
        logger.error(f"DB Error: {e}")
        
def run_ingest():
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/checkpoints", exist_ok=True)
    else:
        files = [f for f in os.listdir("data") if f.endswith(".pdf")]
        if not files:
            print("no pdf")
        else:
            for file in files:
                path = os.path.join("data", file)
                ingestPdf(path)        
  