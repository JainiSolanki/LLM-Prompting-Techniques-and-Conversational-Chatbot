# Import ChatOllama and message schema
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
#     model="llama3.2",
#     base_url="http://localhost:11434",
#     temperature=0.3,
# )


# -----------------------------
# Step 2: Function to extract personal details using goal-driven instruction
# -----------------------------
def get_response_from_bot(user_input):
    messages = [
        SystemMessage(
            content="""
            Goal: Extract personal details from user input.
            
            Instructions: 
                - Return the data as a valid JSON object with keys: name, dob in mm/dd/yyyy format, location. 
                - Do not include extra text or explanation.
            Language: 
                - Return JSON output. Follow the below schema :
                {
                    "name":<string or empty>,
                    "dob":<string in mm/dd/yyyy format or empty>,
                    "location":<string or empty>
                }
            """
            # 🧠 This version follows a clear **instructional structure**:
            # - Goal: What the assistant should achieve
            # - Instructions: Format and rules
            # - Language: Output formatting constraint
            # This improves model compliance for structured outputs
        ),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages)
    return response.content


# -----------------------------
# Step 3: Run the function with a test case
# -----------------------------
user_input = "Hi, I’m Kalind. I was born on 22nd December 1999."

print("GAIL METHOD OUTPUT:\n")
print(get_response_from_bot(user_input))
