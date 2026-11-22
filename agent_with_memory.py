
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from pprint import pprint
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

agent = create_agent(
    "gpt-5",
    checkpointer=InMemorySaver(),

)
message = HumanMessage(content="Hello my name is sean and my favourite color is red")

config = {"configurable":{"thread_id":"1"}}

response = agent.invoke(
    {"messages": [message]},
    config
)

pprint(response)