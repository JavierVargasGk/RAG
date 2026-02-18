import fitz
from db import get_connection_string, file_exists
import time
import voyageai
import psycopg
from dotenv import load_dotenv
import pickle
import os

def getTextFromPDF(filePath: str):
    doc = fitz.open(filePath)
    pages_data = []
    for i, page in enumerate(doc):
        text = page.get_text("text") 
        if text:
            # Clean NUL bytes and normalize weird ligatures (fi, fl, etc.)
            clean_text = text.replace('\x00', '').replace('\ufb01', 'fi').replace('\ufb02', 'fl')
            pages_data.append({"text": clean_text, "page": i + 1})
    doc.close()
    return pages_data

#Split data into chunks of 1000 characters with an overlap of 200 characters (or changed if needed)
def getChunks(data: str,chunkSize: int = 1000, overlap: int=200):
    if(data == ""):
        raise ValueError("Data cannot be empty.")
    if(chunkSize <= 0):
        raise ValueError("Chunk size must be greater than 0.")
    if(overlap < 0):
        raise ValueError("Overlap must be non-negative.")
    if(overlap >= chunkSize):
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
    checkpoint_path = f"checkpoint_{filename}.pkl" # 
    if (file_exists(filename)):
        print(f"Skipping {filename}: Already exists.")
        return
    # i hate everything, 90k tokens gone
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
            all_chunks = data["chunks"]
            all_vectors = data["vectors"]
            all_metadata = data["metadata"]     
    else:
        pages_data = getTextFromPDF(filePath)
        all_chunks = []
        all_metadata = [] 

        for page in pages_data:
            page_chunks = getChunks(page["text"])
            for chunk in page_chunks:
                all_chunks.append(chunk)
                all_metadata.append({"file": filename, "page": page["page"]})

        all_vectors = []
        SMALL_BATCH_SIZE = 15
        print(f"Embedding {len(all_chunks)} chunks from {filename}...")

        for i in range(0, len(all_chunks), SMALL_BATCH_SIZE):
            batch = all_chunks[i : i + SMALL_BATCH_SIZE]
            try:
                res = vo.embed(batch, model="voyage-finance-2", input_type="document")
                all_vectors.extend(res.embeddings)
                print(f"Progress: {i + len(batch)}/{len(all_chunks)} chunks...")
                if i + SMALL_BATCH_SIZE < len(all_chunks):
                    time.sleep(22)
            except voyageai.error.RateLimitError:
                print("Limit")
                time.sleep(60)


        with open(checkpoint_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "vectors": all_vectors, "metadata": all_metadata}, f)

    # DB
    conn_str = get_connection_string()
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                sql = "COPY doc_chunks (content, embedding, filename, page_number) FROM STDIN"
                with cur.copy(sql) as copy:
                    for chunk, vector, meta in zip(all_chunks, all_vectors, all_metadata):
                        clean_chunk = chunk.replace('\x00', '')
                        vector_str = "[" + ",".join(map(str, vector)) + "]"
                        copy.write_row((clean_chunk, vector_str, meta["file"], meta["page"]))
                        
        print(f"Successfully ingested {len(all_chunks)} chunks.")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            
    except Exception as e:
        print(f"Error en DB: {e}.")
        
        
        
if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    else:
        files = [f for f in os.listdir("data") if f.endswith(".pdf")]
        if not files:
            print("no pdf")
        else:
            for file in files:
                path = os.path.join("data", file)
                ingestPdf(path)