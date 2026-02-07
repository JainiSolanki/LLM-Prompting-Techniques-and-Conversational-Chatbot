# Import ChatOllama for model interaction and message classes for structured conversation
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

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
# Map tool names to callables
tools_by_name = {t.name: t for t in tools_list}

llm_with_tools = llm.bind_tools(tools_list)


# -----------------------------
# Step 2: Define function to extract data
# -----------------------------
def get_response_from_bot(user_input):
    print("\n#####################################################\n")
    print(f"User Input: {user_input}")
    print("---------------------------------------")

    messages = [
        SystemMessage(
            content="""You are a helpful assistant. 
Answer the user's question to the best of your ability.
"""
        ),
        HumanMessage(content=user_input),
    ]

    # First pass: propose tool(s) or answer directly
    response = llm_with_tools.invoke(messages)
    print("First pass:")
    print(f"Content: {response.content!r}")
    print(f"Tool Calls: {response.tool_calls}")
    messages.append(response)

    # Execute tool calls if any
    if response.tool_calls:
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            try:
                result = tools_by_name[name].invoke(args)
            except Exception as e:
                result = f"Tool execution error: {e}"
            messages.append(
                ToolMessage(content=str(result), name=name, tool_call_id=tc.get("id"))
            )

        # Second pass: produce final answer
        final = llm_with_tools.invoke(messages)
        print("---------------------------------------")
        print("Final answer after tool execution:")
        print(final.content)
        return final

    # No tools used; probably a free-form answer (e.g., the joke)
    print("---------------------------------------")
    print("Final answer (no tools used):")
    print(response.content)
    return response


get_response_from_bot("What is five added by seven?")


get_response_from_bot(
    "If i had a dozen apples and gave half of them to my team, how many will be left to distribute?"
)
