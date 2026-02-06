# Import the ChatOllama class from langchain_ollama to interface with the Ollama LLM
from langchain_ollama import ChatOllama

from langchain_groq import ChatGroq

# Import pprint to print data structures in a cleaner, more readable format
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

# # Create an instance of the Ollama model with specific configurations
# llm = ChatOllama(
#     model="llama3.2",  # Name of the model you're using (this must match the model available on the Ollama server)
#     base_url="http://localhost:11434",  # URL of the Ollama server (can be local or remote)
#     temperature=0.9,  # Temperature controls randomness (0.0 = deterministic, 1.0 = creative)
# )

# -----------------------------
# Step 2: Create input messages
# -----------------------------

# List of messages to simulate a chat conversation with the LLM
# Each message must have a role: either "user", "assistant", or "system"
# - "user": message from the user
# - "assistant": message from the AI (if you’re continuing a chat)
# - "system": optional message to set behavior/prompt style

messages = [
    {
        "role": "user",  # Role is 'user' since this message is coming from the user
        "content": "My name is Kalind. I am a superhero. I was born on 22nd December, 1999.",
    },
]

# -----------------------------
# Step 3: Send message to the LLM and get response
# -----------------------------

# The 'invoke' method sends the list of messages to the LLM and gets the model's response
response = llm.invoke(messages)

# -----------------------------
# Step 4: Print the results
# -----------------------------

print("Raw response : ")
# This prints the full response object in a clean format using pprint
# Helpful if you want to explore the structure of the response (e.g., metadata, role, content, etc.)
pprint.pprint(response)

print("-------------------------")

# Access just the content (the assistant's message) from the response and print it
print("Assistant response : " + response.content)
