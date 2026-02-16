from pypdf import PdfReader

def getTextFromPDF(filePath: str):
    reader = PdfReader(filePath)
    full_text = []
    for page in reader.pages:
        text += page.extract_text()
        if text:
            full_text.append(text)
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