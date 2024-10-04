from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.llms import Cohere  # Updated import
from langchain_cohere import CohereEmbeddings
import os

# Step 1: Setup Cohere LLM
cohere_api_key = "uUUHn12qUbFiRK1CsKbol45thwnMMIBHt3QIa2xw"  # Replace with your actual Cohere API key
# Alternatively, you can use: cohere_api_key = os.getenv('COHERE_API_KEY')

llm = Cohere(
    model="command-light",
    cohere_api_key=cohere_api_key,
    max_tokens=100
)

# Step 2: Create embeddings using Cohere
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)

# Step 3: Load the FAISS index and create a retriever
# Add the allow_dangerous_deserialization parameter
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retv = db.as_retriever(search_kwargs={"k": 3})

# ... rest of your code remains the same

# Step 4: Function to retrieve and print relevant documents
def get_relevant_documents(query):
    docs = retv.get_relevant_documents(query)
    return docs

def pretty_print_docs(docs):
    print(
        f"\n{'-'*100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# Example usage
docs = get_relevant_documents('Tell us which module is most relevant to deep learning using python')
pretty_print_docs(docs)

for doc in docs:
    print(doc.metadata)

# Step 5: Create a retrieval chain
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv, return_source_documents=True)

# Example usage of the chain
response = chain.invoke("Tell us which module is relevant to deep learning using python")
print(response)