# Import ChatOllama for model interaction and message classes for structured conversation
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
import pprint

# -----------------------------
# Step 1: Initialize the LLM
# -----------------------------
llm = ChatOllama(
    model="llama3.2",  # The name of the model (should be available on Ollama)
    base_url="http://localhost:11434",  # Ollama API endpoint (local or remote)
    temperature=0.3,  # Low temperature for more accurate and predictable results
)


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

llm_with_tools = llm.bind_tools(tools_list)


# -----------------------------
# Step 2: Define function to extract data
# -----------------------------
def get_response_from_bot(user_input):
    print("\n#####################################################\n")

    print(f"User Input: {user_input}")
    print("---------------------------------------")

    # Creating a structured prompt using a system instruction and user message
    messages = [
        SystemMessage(
            content="""You are a helpful assistant. 
Answer the user's question to the best of your ability.
"""
        ),
        HumanMessage(content=user_input),  # The actual user-provided input text
    ]

    # Call the LLM with the messages and return only the content part of the response
    response = llm_with_tools.invoke(messages)

    print("Bot Response:")
    pprint.pprint(response)

    print("---------------------------------------")
    print(f"Content: {response.content}")
    print("---------------------------------------")
    print(f"Tool Calls: {response.tool_calls}")

    return response


get_response_from_bot("What is five added by seven?")


get_response_from_bot(
    "If i had a dozen apples and gave half of them to my team, how many will be left to distribute?"
)


get_response_from_bot("Tell me a math joke.")
