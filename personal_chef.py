from tavily import TavilyClient
from typing import Dict, Any
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from pprint import pprint
from langchain.tools import tool
import base64
load_dotenv()

tavily_client = TavilyClient()

@tool("web_search", description="search the web for information")
def web_search(query:str) -> Dict[str, Any]:
    """search the web for information"""
    return tavily_client.search(query)

personal_chef_agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt="You are a personal chef agent who is responsible for deciding healthy and nutritious dishes for a customer who can provide his or her fridge storage photo with all ingredients he or she have in home. You have to suggest dishes for meal strictly with the ingredient available. You MUST response in format as dish name, ingredients with quantity, brief description on how to prepare the dish"

)

# 1. Read the local image in the same folder as raw bytes
image_path = "/Users/shivamupasane/Downloads/WhatsApp Image 2026-11-26 at 1.25.53 AM.jpeg"  # Replace with your exact PNG filename
with open(image_path, "rb") as image_file:
    img_bytes = image_file.read()

# 2. Base64 encode the bytes
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# 3. Build the multimodal message
multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me what are possible dishes those can be prepared based on ingredients in my fridge"},
    {"type": "image", "base64": img_b64, "mime_type": "image/jpeg"}
])

response = personal_chef_agent.invoke({
    "messages": [multimodal_question]
})

pprint(response['messages'][-1].text)