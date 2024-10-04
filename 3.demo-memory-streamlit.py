from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere  # Changed import statement
import streamlit as st
import os

# Step 1: Set your Cohere API key
cohere_api_key = "uUUHn12qUbFiRK1CsKbol45thwnMMIBHt3QIa2xw"

# Alternatively, you can store the API key in an environment variable and retrieve it like this:
# cohere_api_key = os.getenv('COHERE_API_KEY')

# Step 2: Setup Cohere's Generative AI model
llm = Cohere(
    model="command-light",
    max_tokens=100,  # Adjust max_tokens if needed
    cohere_api_key=cohere_api_key
)

# Step 3: Create a history with a key "chat_messages"
# This history will store messages in Streamlit's session state at the specified key
# StreamlitChatMessageHistory will NOT be persisted or shared across Streamlit sessions
history = StreamlitChatMessageHistory(key="chat_messages")

# Step 4: Create a memory object
memory = ConversationBufferMemory(chat_memory=history)

# Step 5: Create a prompt template
template = "You are an AI chatbot having a conversation with a human.\nHuman: {human_input}\nAI:"
prompt = PromptTemplate(input_variables=["human_input"], template=template)

# Step 6: Create a chain
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Step 7: Use Streamlit to print all messages in the memory and create text input
st.title("Welcome to the ChatBot")

# Display the chat history using Streamlit's chat_message method
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# Get user input and add it to the chat history
if user_input := st.chat_input():
    st.chat_message("human").write(user_input)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    response = llm_chain.run({"human_input": user_input})

    # Display AI response
    st.chat_message("ai").write(response)
