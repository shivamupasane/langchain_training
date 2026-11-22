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



#create agent

from langchain.agents import create_agent
agent = create_agent(model=model, tools=[])
from langchain.messages import HumanMessage
message = HumanMessage(content="What's the capital of the Moon?")
response = agent.invoke({
    "messages": [message]   
})
pprint(response)
#print(response['messages'][-1].content)