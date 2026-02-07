from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def count_words(text: str) -> int:
    """Count the number of words in the input string."""
    return len(text.split())

@tool
def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()

@tool
def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

tools_list = [count_words, lowercase, uppercase]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

system_prompt = """
You are a smart and beautiful AI assistant.
Rules:
- Use tools ONLY when required.
- If text is uppercase → convert to lowercase.
- If text is lowercase → convert to uppercase.
- Use count_words when asked about number of words.
- Otherwise answer normally.
"""

agent = create_agent(
    model=llm,
    tools=tools_list,
    system_prompt=system_prompt,
)

def chat():
    print("Agent Ready! Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = agent.invoke({
            "messages": [
                {"role": "user", "content": user_input}
            ]
        })

        print("Agent:", response["messages"][-1].content)
        print("--------------------------------------------------")

chat()
