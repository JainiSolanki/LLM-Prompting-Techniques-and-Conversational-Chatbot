# Import ChatOllama for model interaction and message classes for structured conversation
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Step 1: Initialize the LLM model
# -----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9,
)

# llm = ChatOllama(
#     model="llama3.2",  # The name of the model (should be available on Ollama)
#     base_url="http://localhost:11434",  # Ollama API endpoint (local or remote)
#     temperature=0.3,  # Low temperature for more accurate and predictable results
# )


# -----------------------------
# Step 2: Define function to extract data
# -----------------------------
def get_response_from_bot(user_input):
    # Creating a structured prompt using a system instruction and user message
    messages = [
        SystemMessage(
            content="""You are an information extractor. 
Extract the person's name, date of birth in mm/dd/yyyy format, and location from the user input.
Return it as a valid JSON."""
            # 🧠 This prompt tells the model *what* to do (extract info), *how* to format it (mm/dd/yyyy), and *what to return* (JSON)
        ),
        HumanMessage(content=user_input),  # The actual user-provided input text
    ]

    # Call the LLM with the messages and return only the content part of the response
    response = llm.invoke(messages)
    return response.content


# -----------------------------
# Step 3: Run the function with test input
# -----------------------------
user_input = "Hi, I’m Kalind. I was born on 22nd December 1999 in Jamnagar."

print("ZERO-SHOT OUTPUT:\n")
print(get_response_from_bot(user_input))
