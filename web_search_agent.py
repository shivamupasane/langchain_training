#agents are different from normal chatbot as they take actions and preserve output from that actions

from tavily import TavilyClient
from typing import Dict, Any
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from pprint import pprint
from langchain.tools import tool
load_dotenv()

tavily_client = TavilyClient()

@tool("web_search", description="search the web for information")
def web_search(query:str) -> Dict[str, Any]:
    """search the web for information"""
    return tavily_client.search(query)

websearch_agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt="You are a web search agent"
)
question = HumanMessage(content="Who is current mayor of San Fransisco?")
response = websearch_agent.invoke({
    "messages": [question]
})
pprint(response['messages'][1].tool_calls)
pprint(response['messages'][-1].text)