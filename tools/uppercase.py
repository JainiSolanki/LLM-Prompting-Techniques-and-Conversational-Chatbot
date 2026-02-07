from langchain_core.tools import tool
@tool
def uppercase(text: str) -> str:
    """Convert the text to uppercase."""
    return text.upper()

print(uppercase.run({"text": "Jaini Solanki"}))  #always provide dictionary as input while calling