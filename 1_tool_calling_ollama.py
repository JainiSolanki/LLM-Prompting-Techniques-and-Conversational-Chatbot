from ollama import Client
from pprint import pprint


# Assuming Ollama server is running on a different host and port
client = Client(host="http://localhost:11434")


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


def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    return int(a) - int(b)


# Tools can still be manually defined and passed into chat
subtract_two_numbers_tool = {
    "type": "function",
    "function": {
        "name": "subtract_two_numbers",
        "description": "Subtract two numbers",
        "parameters": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number"},
            },
        },
    },
}


# Tools can still be manually defined and passed into chat
add_two_numbers_tool = {
    "type": "function",
    "function": {
        "name": "add_two_numbers",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number"},
            },
        },
    },
}


def print_ollama_response(user_input):
    response = client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": user_input}],
        # provide a weather checking tool to the model
        tools=[add_two_numbers, subtract_two_numbers_tool],
    )

    print("Full response:")
    pprint(response)

    print("---------------------------------------")

    print("Model response:")
    print(response["message"].content)

    print("---------------------------------------")

    print("Tool calls:")
    pprint(response["message"].tool_calls)


print_ollama_response("What is three added by two?")


print_ollama_response(
    "If i have ten coins and give three of them to my friend, how many do I have now?"
)
