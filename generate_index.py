import os
import pickle
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

def chunk_text(text, max_length=250):
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_length:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current: chunks.append(current.strip())
    return chunks

def load_documents(base_path="compagnies"):
    docs, tags = [], []
    for company in os.listdir(base_path):
        path = os.path.join(base_path, company)
        if not os.path.isdir(path): continue
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if filename.endswith(".pdf"):
                reader = PdfReader(full_path)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif filename.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                continue
            docs.append(text)
            tags.append(company)
    return docs, tags

docs, tags = load_documents()
chunks, chunk_tags = [], []
for doc, tag in zip(docs, tags):
    for chunk in chunk_text(doc):
        chunks.append(chunk)
        chunk_tags.append(tag)

encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
embeddings = encoder.encode(chunks, show_progress_bar=True)

with open("chunks.pkl", "wb") as f:
    pickle.dump({i: (chunks[i], chunk_tags[i]) for i in range(len(chunks))}, f)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index.bin")

print("✅ Indexation terminée.")
