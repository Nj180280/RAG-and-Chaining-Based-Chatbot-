from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.llms import Cohere  # Changed import statement
import os

# Step 1: Set your Cohere API key
# Ensure the API key is passed correctly
cohere_api_key = "uUUHn12qUbFiRK1CsKbol45thwnMMIBHt3QIa2xw"

# Alternatively, you can store the API key in an environment variable and retrieve it like this:
# cohere_api_key = os.getenv('COHERE_API_KEY')

# Step 2: Setup Cohere's Generative AI model
llm = Cohere(
    model="command-light",
    max_tokens=100,  # Adjust max_tokens if needed
    cohere_api_key=cohere_api_key
)

# Step 3: Create the prompt template
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot who explains in steps."
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Step 4: Create memory to remember the chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

# Step 5: Create a conversation chain using LLMChain and memory
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=summary_memory
)

# Step 6: Invoke the chain with a question
response = conversation.invoke({"question": "What is the capital of India?"})
print("Response:", response)

# Step 6b: Print all messages in the memory
print("Chat History:")
print(memory.chat_memory.messages)

print("\nSummary of the conversation is: " + summary_memory.buffer)

# Step 7: Ask another question
response = conversation.invoke({"question": "What is OCI data science certification?"})
print("Response:", response)

# Step 8: Print all messages in the memory again to see the updated conversation
print("\nUpdated Chat History:")
print(memory.chat_memory.messages)

print("\nSummary of the conversation is: " + summary_memory.buffer)
