# Import the ChatOllama class to interact with an Ollama-powered LLM
from langchain_ollama import ChatOllama

from langchain_groq import ChatGroq

# Import pprint to print Python data structures in a readable format
import pprint

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Step 1: Initialize the LLM model
# -----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9,
)

# # Initialize the ChatOllama instance with desired model settings
# llm = ChatOllama(
#     model="llama3.2",  # Name of the LLM model to use (must be available on your Ollama server)
#     base_url="http://localhost:11434",  # URL of the running Ollama server
#     temperature=0.5,  # Medium temperature: some creativity, but still relatively stable
# )

# -----------------------------
# Step 2: Define the conversation messages
# -----------------------------

# This is a multi-turn message format using the OpenAI-style chat structure.
# The model will follow the system instruction and then respond to the user input.

messages = [
    {
        "role": "system",  # System messages help guide the behavior of the assistant
        "content": "You are a helpful assistant. Convert the user input from English to French.",
    },
    {
        "role": "user",  # User message is what the person (you) sends to the assistant
        "content": "My name is Kalind. I am from Jamnagar in Gujarat. I was born on 22nd December, 1999.",
    },
]

# -----------------------------
# Step 3: Send the messages to the model and get the response
# -----------------------------

# Invoke the model by passing in the chat messages
# The assistant will respond based on the system instruction
response = llm.invoke(messages)

# -----------------------------
# Step 4: Print the model's output
# -----------------------------

print("Raw response : ")
# Use pprint to inspect the full response object returned by the model
pprint.pprint(response)

print("-------------------------")

# Extract and display only the assistant's message from the response
print("Assistant response : " + response.content)
