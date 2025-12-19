from dotenv import load_dotenv
from pprint import pprint
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from pydantic import BaseModel
load_dotenv()

system_prompt = "Você é um chef de cozinha que faz receitas de comida. Eu direi nomes de pratos, e você dará a receita"

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    instructions: list[str]

model = init_chat_model(model="gpt-4o-mini", temperature=1.0)
agent = create_agent(
    model=model, 
    system_prompt=system_prompt,
    response_format=Recipe,
    tools=[]
)

query = "Bolo de chocolate"
response = agent.invoke({
    "messages": [
        HumanMessage(content=query)
    ]
})

pprint(response)

