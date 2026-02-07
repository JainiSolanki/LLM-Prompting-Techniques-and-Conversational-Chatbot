from ollama import Client
from pprint import pprint

# Assuming Ollama server is running on a different host and port
client = Client(host="http://localhost:11434")


def print_ollama_respone(user_input):
    response = client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": user_input}],
    )
    pprint(response)

    print("---------------------------------------")

    pprint(response["message"].content)


print_ollama_respone("what is two plus two?")
