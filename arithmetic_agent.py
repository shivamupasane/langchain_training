#tools

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from pprint import pprint
load_dotenv()
#normal tool is defined as below


# @tool
# def square_root(x: Float) -> float:
#     """calculate the square root of a number"""
#     return x ** 0.5


# how to define a tool
#1. use @tool decorator
#2. name of tool and desacription should be clear
#3. by default -> name of function --> name of tool and docString --> description of function --> description of tool

#we can customise tool name and tool description as below

@tool("square_root", description="calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5

#how agent will invoke this tool    
#tool1.invoke({"x":467})

arithmetic_agent = create_agent(
    model="gpt-5-nano",
    tools=[tool1],
    system_prompt= "You are a arithmetic wizard. Use your tools to calculate the square root and square of any number"
)
question = HumanMessage(content="What is square root of 467?")
response = arithmetic_agent.invoke({
    "messages": [question]
})
pprint(response['messages'][1].tool_calls)
