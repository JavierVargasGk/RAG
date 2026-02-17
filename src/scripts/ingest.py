from pypdf import PdfReader
from db import get_connection_string, file_exists
import voyageai
import psycopg
from dotenv import load_dotenv
import os

def getTextFromPDF(filePath: str):
    reader = PdfReader(filePath)
    pages_data = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_data.append({"text": text, "page": i + 1})
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
    
    if (file_exists(filename)):
        print(f"Skipping {filename}: Already exists.")
        return
    
    pages_data = getTextFromPDF(filePath)
    all_chunks = []
    all_metadata = [] 

    for page in pages_data:
        page_chunks = getChunks(page["text"])
        for chunk in page_chunks:
            all_chunks.append(chunk)
            all_metadata.append({"file": filename, "page": page["page"]})

    all_vectors = []
    print(f"Embedding {len(all_chunks)} chunks from {filename}...")
    for batch in makeBatches(all_chunks):
        res = vo.embed(
            batch, 
            model="voyage-finance-2", 
            input_type="document"
        )
        all_vectors.extend(res.embeddings)

    # 4. DB 
    conn_str = get_connection_string()
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                sql = "COPY doc_chunks (content, embedding, filename, page_number) FROM STDIN"
                with cur.copy(sql) as copy:
                    for chunk, vector, meta in zip(all_chunks, all_vectors, all_metadata):
                        vector_str = "[" + ",".join(map(str, vector)) + "]"
                        copy.write_row((chunk, vector_str, meta["file"], meta["page"]))
                        
        print(f"Successfully ingested {len(all_chunks)} chunks from {filename}")
    except Exception as e:
        print(f"Error during DB ingestion: {e}")
        
        
        
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