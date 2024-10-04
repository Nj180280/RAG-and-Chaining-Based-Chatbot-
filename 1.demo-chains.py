from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
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
    max_tokens=200,
    cohere_api_key=cohere_api_key
)

# Step 3: Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a very knowledgeable scientist who provides accurate and eloquent answers to scientific questions."),
    ("human", "{question}")
])

# Step 4: Create a chain using LLMChain and invoke it
# Legacy chain
chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
response = chain.invoke({"question": "What are the basic elements of matter?"})
print("Response from legacy chain:")
print(response)

# Step 5: Use Langchain expression language to compose a direct chain and invoke it
# Direct chain
runnable = prompt | llm | StrOutputParser()
response = runnable.invoke({"question": "What are the basic elements of matter?"})
print("Response from Direct Chain:")
print(response)