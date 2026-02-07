from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()


@tool
def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
      a (int): The first number
      b (int): The second number

    Returns:
      int: The sum of the two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    # E.g. this would prevent "what is 30 + 12" to produce '3012' instead of 42
    return int(a) + int(b)


@tool
def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    return int(a) - int(b)


tools_list = [add_two_numbers, subtract_two_numbers]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
)

# llm = ChatOllama(
#     model="llama3.1",
#     base_url="http://localhost:11434",
#     temperature=0.2,
# )

agent_prompt = """
You are a helpful assistant. 
Answer the user's question to the best of your ability.
Give output in 2-3 sentences only.
"""

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


get_response_from_agent("What is five added by seven?")


get_response_from_agent(
    "If i had a dozen apples and gave half of them to my team, how many will be left to distribute?"
)
