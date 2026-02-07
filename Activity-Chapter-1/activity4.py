# Import the ChatOllama class to use an Ollama-based LLM
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
#     model="llama3.2",  # The name of the model (must be available on the Ollama server)
#     base_url="http://localhost:11434",  # URL of the Ollama server (local or remote)
#     temperature=0.5,  # Controls creativity (0 = focused, 1 = creative). 0.5 = balanced
# )

# -----------------------------
# Step 2: Define initial message history
# -----------------------------

# This list keeps track of the entire conversation history
# It starts with a system message that tells the assistant how to behave
messages = [
    SystemMessage(
        content="You are a charming and helpful assistant. Try to help the user with their request."
        # 🧠 This sets the assistant’s tone: friendly, helpful, and engaging
    )
]


# -----------------------------
# Step 3: Define response generation function
# -----------------------------
def get_response_from_bot(user_input):
    # Append the new user input to the conversation
    messages.append(HumanMessage(content=user_input))

    # Invoke the LLM with the current full message history
    response = llm.invoke(messages)

    # Add the model's response to the conversation history
    messages.append(response)

    # Print the number of messages so far (helps monitor growth)
    print("---------------------------------")
    print(f"Messages length: {len(messages)}")

    # Return only the text content of the assistant's reply
    return response.content

def save_chat_to_file():
    with open("chat_log.txt", "w", encoding="utf-8") as file:
        for msg in messages:

            # Skip system message (optional but recommended)
            if isinstance(msg, SystemMessage):
                continue

            if isinstance(msg, HumanMessage):
                file.write(f"User: {msg.content}\n")

            else:  # AI response
                file.write(f"Assistant: {msg.content}\n")

    print("\n✅ Conversation saved to chat_log.txt")

# -----------------------------
# Step 4: Run chatbot loop
# -----------------------------
while True:
    # Get user input
    user_input = input("User: ")

    # If user says "bye", exit the loop
    if "bye" in user_input.lower():
        save_chat_to_file()
        print("Bot: Goodbye!")
        break

    # Generate response and print it
    response = get_response_from_bot(user_input)
    print("Bot: ", response)
