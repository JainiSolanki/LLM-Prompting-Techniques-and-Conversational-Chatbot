from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Step 1: Initialize the LLM model
# -----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9,
)

# llm = ChatOllama(
#     model="llama3.1",
#     base_url="http://localhost:11434",
#     temperature=0.2,
# )

agent_prompt = """You are a helpful assistant. 
Answer the user's question to the best of your ability.
"""

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=agent_prompt,
)


def get_response_from_bot(user_input):
    messages = [
        SystemMessage(content=agent_prompt),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages)
    return response


def get_response_from_agent(user_input):
    input_messages = [
        {"role": "user", "content": user_input},
    ]

    response = agent.invoke({"messages": input_messages})

    return response


question = "What is the capital of France?"

print("User question: ", question)

print("\n---------------------------\n")

bot_response = get_response_from_bot(question)
print("Bot response : ", bot_response)

print("\n---------------------------\n")

agent_response = get_response_from_agent(question)
print("Agent response : ", agent_response)
