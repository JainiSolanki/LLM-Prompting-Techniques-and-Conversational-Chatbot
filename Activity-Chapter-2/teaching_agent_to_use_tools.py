from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

@tool("count_words", return_direct=True)
def count_words(text: str) -> int:
    """Count the number of words in the input String."""
    return len(text.split())

@tool(return_direct=True)
def lowercase(text: str) -> str:
    """Convert the input text to lowercase."""
    return text.lower()

@tool(return_direct=True)
def uppercase(text: str) -> str:
    """Convert the text to uppercase."""
    return text.upper()

tools_list = [uppercase, lowercase, count_words]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

agent_prompt = """
You are a helpful assistant. 
You can perform various operations using the tools provided.
If a user give query in Upper case, convert it to lowercase using the lowercase tool and respond.
If a user give query in Lower case, convert it to Uppercase using the uppercase tool and respond.
When you receive a user query, determine if it requires the use of any of the tools.
If it does, call the appropriate tool with the correct arguments to get the answer.
If the query does not require any tool, answer it directly."""

agent = create_agent(
    model=llm,
    tools=tools_list,
    system_prompt=agent_prompt,
)

def get_response_from_agent(user_input):
    input_messages = [
        {"role": "user", "content": user_input},
    ]

    for step in agent.stream({"messages": input_messages}, stream_mode="values"):
        step["messages"][-1].pretty_print()

    print(
        "---------------------------------------------------------------------------------"
    )

get_response_from_agent("KALIND SARDA")
get_response_from_agent("How many words are in the sentence 'I am learning LangChain' ?")
get_response_from_agent("Hello How are you doing today?")