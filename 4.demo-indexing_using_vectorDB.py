from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

# Step 1: Load PDF documents
pdf_loader = PyPDFDirectoryLoader("./pdf-docs")
loaders = [pdf_loader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

# Step 2: Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1880, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total number of documents: {len(all_documents)}")

# Step 3: Setup Cohere Embeddings
cohere_api_key = "uUUHn12qUbFiRK1CsKbol45thwnMMIBHt3QIa2xw"  # Replace with your actual Cohere API key
# Alternatively, you can use: cohere_api_key = os.getenv('COHERE_API_KEY')

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key,
)

# Step 4: Setup FAISS vector store
db = FAISS.from_documents(all_documents, embeddings)

# Step 5: Use the vector store as a retriever
retriever = db.as_retriever()

# Step 6: Process documents in batches (if needed)
batch_size = 96
num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)

for batch_num in range(num_batches):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, len(all_documents))
    
    batch_documents = all_documents[start_index:end_index]
    # If you need to process each document, you can do it here
    # For example: db.add_documents(batch_documents)
    
    print(f"Processed documents {start_index} to {end_index}")

# Step 7: Save the FAISS index
db.save_local("faiss_index")

print("All documents have been processed and stored in the FAISS vector database.")
print("FAISS index has been saved locally.")

# To demonstrate how to load the index later:
# loaded_db = FAISS.load_local("faiss_index", embeddings)
# print("FAISS index has been loaded successfully.")