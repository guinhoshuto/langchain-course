from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from typing import Dict, Any
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

@tool("web_search")
def web_search(query: str) -> Dict[str, Any]:
    """Retorna uma resposta baseada em uma busca na web"""
    return tavily_client.search(query)

@tool("square_root")
def square_root(x: float) -> float:
    """Retorna a raiz quadrada de um número"""
    return x ** 0.5

@tool("add")
def add(x: float, y: float) -> float:
    """Retorna a soma de dois números"""
    return x + y

@tool("subtract")
def subtract(x: float, y: float) -> float:
    """Retorna a diferença entre dois números"""
    return x - y

agent = create_agent(
    model='gpt-5',
    tools=[square_root, add, subtract, web_search],
)

response = agent.invoke({
    "messages": [
        HumanMessage(content="Persona 3 best persona")
    ]
})

print(response)
print(response["messages"][-1].tool_calls, response["messages"][-1])
