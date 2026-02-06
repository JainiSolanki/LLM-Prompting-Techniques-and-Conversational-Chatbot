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

# -----------------------------
# Step 2: Create a reusable system message
# -----------------------------

system_message = SystemMessage(
    content="You are a charming and helpful assistant. Try to help the user with their request."
)

# Start conversation with system message
messages = [system_message]

# Track ONLY user messages
user_message_count = 0


# -----------------------------
# Step 3: Define response function
# -----------------------------
def get_response_from_bot(user_input):
    global messages
    global user_message_count

    # Add user message
    messages.append(HumanMessage(content=user_input))
    user_message_count += 1

    # MEMORY RESET LOGIC
    if user_message_count > 5:
        print("\n⚠️ Chat history is now too long. Resetting memory...\n")

        # Reset messages
        messages = [system_message]

        # Reset counter
        user_message_count = 1  # Count current message

        # Add the current message again
        messages.append(HumanMessage(content=user_input))

    # Invoke model
    response = llm.invoke(messages)

    # Save response
    messages.append(response)

    print("---------------------------------")
    print(f"Total messages stored: {len(messages)}")
    print(f"User messages count: {user_message_count}")

    return response.content


# -----------------------------
# Step 4: Chat loop
# -----------------------------
while True:
    user_input = input("User: ")

    if "bye" in user_input.lower():
        print("Bot: Goodbye!")
        break

    response = get_response_from_bot(user_input)
    print("Bot:", response)
