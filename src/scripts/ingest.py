from pypdf import PdfReader
import voyageai
import psycopg
from dotenv import load_dotenv
import os

def getTextFromPDF(filePath: str):
    reader = PdfReader(filePath)
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)
    return "\n".join(full_text)

def getText(filePath: str):
    with open(filePath, 'r', encoding='utf-8') as file:
        data = file.read()
    return data
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

def get_connection_string():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"



def ingestPdf(filePath: str):
    text = getTextFromPDF(filePath)
    chunks = getChunks(text)
    all_vectors = []
    print(f"Embedding {len(chunks)} chunks in batches...")
    for batch in makeBatches(chunks):
        res = vo.embed(
            batch, 
            model="voyage-finance-2", 
            input_type="document"
        )
        all_vectors.extend(res.embeddings)

    # DB 
    conn_str = get_connection_string()
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                with cur.copy("COPY doc_chunks (content, embedding, filename) FROM STDIN") as copy:
                    for chunk, vector in zip(chunks, all_vectors):
                        copy.write_row((chunk, vector, os.path.basename(filePath)))
        print(f"Successfully ingested {len(chunks)} chunks from {filePath}")
    except Exception as e:
        print(f"Error: {e}")
        
        
        
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