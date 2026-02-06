# Import required components to work with ChatOllama and structured chat messages
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
#     model="llama3.2",  # LLM model name (must be downloaded/running on Ollama)
#     base_url="http://localhost:11434",  # Server URL for the Ollama API (local or remote)
#     temperature=0.5,  # Medium creativity level for friendly, engaging responses
# )


# -----------------------------
# Step 2: Define a function to get assistant's response
# -----------------------------
def get_response_from_bot(user_input):
    # Creating the message history (system prompt + user input)
    messages = [
        SystemMessage(
            content="You are a charming and helpful assistant. Try to help the user with their request."
            # 🧠 This system message sets the assistant's tone (charming + helpful) and tells it to assist
        ),
        HumanMessage(content=user_input),  # Wraps user input in a structured format
    ]

    # Send the messages to the model and get the response
    response = llm.invoke(messages)
    return response.content  # Extract the text content of the assistant's reply


# -----------------------------
# Step 3: Run the chatbot loop
# -----------------------------
while True:
    # Prompt the user for input
    user_input = input("User: ")

    # Exit condition — if the user types "bye", end the loop
    if "bye" in user_input.lower():
        print("Bot: Goodbye!")
        break

    # Get model-generated response based on input
    response = get_response_from_bot(user_input)

    # Display the assistant's reply
    print("Bot: ", response)
