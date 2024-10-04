import os
import json
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Cohere
from langchain_cohere import CohereEmbeddings
import streamlit as st

# Hardcoded API Key for Cohere
cohere_api_key = "uUUHn12qUbFiRK1CsKbol45thwnMMIBHt3QIa2xw"  # Replace with your actual API key

# File to store questions
QUESTIONS_FILE = "questions.json"

# Load previous questions if file exists
def load_questions():
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, "r") as file:
            return json.load(file)
    return []

# Save questions to file
def save_questions(questions):
    with open(QUESTIONS_FILE, "w") as file:
        json.dump(questions, file)

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
        max_tokens=300  # Increase tokens for more detailed responses
    )
    
    # Initialize ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
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
    # Improved prompt design
    bot_response = chain({"question": user_message})
    return bot_response

# Step 4 - Streamlit UI
st.set_page_config(page_title="RAG Based Chatbot", page_icon="🤖")
st.title("RAG Based Chatbot powered by Cohere")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for questions
if "questions" not in st.session_state:
    st.session_state.questions = load_questions()

# Streamlit chat input
user_input = st.text_input("Ask a question about AI and Deep Learning:")

# Process user input and get bot response
if st.button("Submit") and user_input:
    with st.spinner("Fetching response..."):
        bot_response = chat(user_input)
        # Clear previous messages and append only the current question and its response
        st.session_state.messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": bot_response}
        ]
        
        # Append the question to the session state and save it
        st.session_state.questions.append(user_input)
        save_questions(st.session_state.questions)

# Display the most recent chat message in the main area
if st.session_state.messages:
    message = st.session_state.messages[-1]  # Get the last message
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.write("Answer: ", message["content"]["answer"])  # Display the answer part
            st.write("References:")
            for doc in message["content"]["source_documents"]:
                st.write(f"- {doc.metadata['source']} (Page: {doc.metadata['page']})")

# Streamlit sidebar for question history
with st.sidebar:
    st.title("Conversation History")
    for i, question in enumerate(st.session_state.questions):
        st.write(f"**Q{i + 1}:** {question}")  # Show only the questions

    st.info(
        "This chatbot uses FAISS for retrieval and Cohere for question answering. "
        "It's designed to answer questions about AI courses."
    )
