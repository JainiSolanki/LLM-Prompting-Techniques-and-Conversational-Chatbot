from langchain_core.tools import tool

@tool("Count Words", return_direct=True)
def count_words(text : str) -> int:
    """Count the number of words in the input String."""
    return len(text.split())

print(count_words.run({"text": "Jaini from Charusat University"})) 