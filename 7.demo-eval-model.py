import os
from uuid import uuid4
import langsmith
from langchain import smith
from langchain.smith import RunEvalConfig
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Cohere

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Your LangChain API key

cohere_api_key = "uUUHn12qUbFiRK1CsKbol45thwnMMIBHt3QIa2xw"  # Your Cohere API key

# Setup Cohere LLM
llm = Cohere(
    model="command",
    cohere_api_key=cohere_api_key,
    max_tokens=400
)

# Setup Cohere Embeddings
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)

# Load the index and create a retriever
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retv = db.as_retriever(search_kwargs={"k": 8})

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv)

# Define the evaluators
eval_config = RunEvalConfig(
    evaluators=[
        "qa",
        RunEvalConfig.Criteria("relevance")
    ],
    custom_evaluators=[],
    eval_llm=llm
)

client = langsmith.Client()
try:
    chain_results = client.run_on_dataset(
        dataset_name="AIFoundationsDS-49cb4719",
        llm_or_chain_factory=chain,
        evaluation=eval_config,
        concurrency_level=5,
        verbose=True,
    )
    print("Evaluation completed. Results stored in LangSmith.")
    print(f"Dataset: {chain_results.dataset_name}")
    print(f"Number of examples: {chain_results.num_examples}")
    print(f"Results: {chain_results.results}")
except Exception as e:
    print(f"An error occurred during evaluation: {str(e)}")