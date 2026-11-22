from dotenv import load_dotenv
from pprint import pprint
from langchain.chat_models import init_chat_model
load_dotenv()

#higher temparature means more randomness in the output, lower temperature means more deterministic output

model = init_chat_model(model="gpt-5-nano", use_responses_api=True, temperature=5.0)
#response = model.invoke("What's the capital of the Moon?")

#response
#print(response.content)


#pprint(response.response_metadata)


# to use other models, you can use the following code:
#https://docs.langchain.com/oss/python/integrations/chat
#model = init_chat_model(model="claude-sonnet-4-6")

#response = model.invoke("What's the capital of the Moon?")
#print(response.content)

#for google gemini models, you can use the following code:
#from langchain_google_genai import ChatGoogleGenerativeAI

#model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

#response = model.invoke("What's the capital of the Moon?")
#print(response.content)





# to create agent around our model
#from langchain.agents import create_agent

#agent = create_agent(model=model)

#other way to pass model name
#agent = create_agent(model="claude-sonnet-4-5")
#agent = create_agent("gpt-5-nano")

#create agent

from langchain.agents import create_agent
agent = create_agent(model=model, tools=[])
from langchain.messages import HumanMessage, AIMessage
message = HumanMessage(content="What's the capital of the Moon?")

#response without streaming
# response = agent.invoke({
#     "messages": [HumanMessage(content="What's the capital of the Moon?"), AIMessage(content="The capital of Moon is Luna city."), HumanMessage(content="Interesting, tell me more about Luna city")]   
# })
#pprint(response)
#pprint(response['messages'][-1].text)

#response with streaming
# for token,metadata in agent.stream(
#     {
#         "messages": [HumanMessage(content="tell me more about Luna city")]
#     },
#     stream_mode="messages"
# ):
#     if token.text:
#         print(token.text,end="",flush=True)




#prompt engineering
# 1. system_prompt
# 2. system_promt - with example - with response format
# 3. response_format as a python class (Model) so we can access key directly

system_prompt = "You are a science fictional writer, create a capital city at the users requests."
system_prompt_with_example = """
You are a science fictional writer, create a capital city at the users requests.

User:What is capital of mars?
Scifi Writer: Marsialis

User:What is capital of Venus?
Scifi writer: Venusovia

"""
system_prompt_with_structure = """

You are a science fictional writer, create a capital city at the users requests.
Please keep to the below structure.
Name: The name of capital city
Location: Where it is based
Vibe: 2-3 words to describe its vibe
Economy: Main industries

"""
from pydantic import BaseModel
class CapitalInfo(BaseModel):
    name:str
    location:str
    vibe:str
    economy:str
    
scifi_agent = create_agent(
    model="gpt-5-nano",
    system_prompt=system_prompt,
    response_format=CapitalInfo
)
response = scifi_agent.invoke({
    "messages": [message]
})
#print(response['messages'][-1].text)
print(response['structured_response'].name)
