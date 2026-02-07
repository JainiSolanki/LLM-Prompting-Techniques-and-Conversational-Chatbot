from groq import Groq
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

client = Groq()


def print_groq_respone(user_input):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            },
        ],
        # The language model which will generate the completion.
        model="llama-3.1-8b-instant",
    )
    pprint(response)

    print("---------------------------------------")

    pprint(response.choices[0].message.content)


print_groq_respone("what is two plus two?")
