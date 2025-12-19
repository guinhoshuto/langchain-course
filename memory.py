from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

agent = create_agent("gpt-5", checkpointer=MemorySaver())
question = HumanMessage(content="Qual é a capital do Brasil?")

response = agent.invoke(
    {"messages": [question]},
    {"configurable": {"thread_id": "1"}}
)

question2 = HumanMessage(content="Mas qual foi a primeira??")
response2 = agent.invoke(
    {"messages": [question2]},
    {"configurable": {"thread_id": "1"}}
)

print(response2)
