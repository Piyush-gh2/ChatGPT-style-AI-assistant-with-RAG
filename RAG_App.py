import os
import faiss
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

folder = "data"
documents = []

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder, file))
        docs = loader.load()
        documents.extend(docs)

print("Documents loaded:", len(documents))

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

texts = [chunk.page_content for chunk in chunks]

# Create embeddings
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Ask question
query = input("Ask a question: ")

query_vector = model.encode([query])
query_vector = np.array(query_vector).astype("float32")

D, I = index.search(query_vector, k=3)

results = [texts[i] for i in I[0]]

print("\nAnswer:\n")
print(" ".join(results))