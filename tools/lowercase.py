from langchain_core.tools import tool
@tool("lowercase", return_direct=True)
def lowercase(text: str) -> str:
    """Convert the input text to lowercase."""
    return text.lower()

print(lowercase.run({"text": "Jaini Solanki"})) #always provide dictionary as input while calling