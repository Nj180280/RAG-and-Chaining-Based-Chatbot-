import faiss
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Cohere
from langchain_cohere import CohereEmbeddings
import streamlit as st

# Hardcoded API Key for Cohere
cohere_api_key = "uUUHn12qUbFiRK1CsKbol45thwnMMIBHt3QIa2xw"  # Replace with your actual API key

# Step 1 - Setup the conversational retrieval chain
def create_chain():
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key
    )
    
    # Load FAISS index with deserialization allowed
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retv = db.as_retriever(search_kwargs={"k": 4})
    
    # Set up Cohere LLM
    llm = Cohere(
        model="command",
        cohere_api_key=cohere_api_key,
        max_tokens=1000
    )
    
    # Initialize ConversationBufferMemory
    memory = ConversationSummaryMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer', 
        llm=llm
    )
    
    # Create the conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=retv, 
        memory=memory,
        return_source_documents=True
    )
    
    return qa

# Step 2 - Create the chain
chain = create_chain()

# Step 3 - Define chat function
def chat(user_message):
    bot_response = chain({"question": user_message})
    return bot_response

# Step 4 - Streamlit UI
st.set_page_config(page_title="AI Course Chatbot", page_icon="🤖")
st.subheader("AI Course Chatbot powered by Cohere")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit chat input
user_input = st.chat_input("Ask a question about AI and Deep Learning:")

# Process user input and get bot response
if user_input:
    bot_response = chat(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.write("Answer: ", message["content"]["answer"])
            st.write("References:")
            for doc in message["content"]["source_documents"]:
                st.write(f"- {doc.metadata['source']} (Page: {doc.metadata['page']})")

# Streamlit sidebar
if __name__ == "__main__":
    st.sidebar.title("About")
    st.sidebar.info(
        "This chatbot uses FAISS for retrieval and Cohere for question answering. "
        "It's designed to answer questions about AI courses."
    )
