# Import necessary classes
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
# Step 2: Define function to extract data with few-shot examples
# -----------------------------
def get_response_from_bot(user_input):
    messages = [
        SystemMessage(
            content="""You are an information extractor. 
Below are examples of how to extract a person's name, date of birth in mm/dd/yyyy format, and location from text, and return a valid JSON.

Example 1:
Input: "Hi, I’m Anya. I was born on July 4th, 1995 in Goa."
Output: {"name": "Anya", "dob": "07/04/1995", "location": "Goa"}

Example 2:
Input: "My name is Rohan. I was born on 01-01-2001."
Output: {"name": "Rohan", "dob": "01/01/2001", "location": ""}

Now extract information from the input below and return only the JSON.
"""
            # 🧠 This is a **few-shot** prompt:
            # You provide **examples** of correct input-output format
            # LLMs tend to follow patterns better when examples are shown
        ),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages)
    return response.content


# -----------------------------
# Step 3: Run the function with a test input
# -----------------------------
user_input = "Hi, I’m Kalind. I was born on 22nd December 1999."

print("FEW-SHOT OUTPUT:\n")
print(get_response_from_bot(user_input))
